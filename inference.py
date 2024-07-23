import fire
import torch
import torchaudio
import numpy as np

from jiwer import wer as calculate_wer
from whisper_normalizer.basic import BasicTextNormalizer
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperProcessor,
)

language = "uk"
normalizer = BasicTextNormalizer()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch_dtype = torch.float16
torch_dtype = torch.float32


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inference(
    MODEL="openai/whisper-large-v3",
    DATASET="chime4",
    TEST_DATA="",
    CKPT="",
):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL)
    processor = WhisperProcessor.from_pretrained(
        MODEL, language=language, task="transcribe"
    )
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    def data_preparation(data_path, feature_extractor):
        with open(data_path + "wav.scp", "r") as f1:
            wave_data = f1.readlines()
        with open(data_path + "text", "r") as f2:
            trans_data = f2.readlines()

        audio_data, txt_data = [], []
        for i in range(len(wave_data)):
            audio_data.append(wave_data[i])
            txt_data.append(trans_data[i])

        audio_dataset = []
        for audio_line, text_line in zip(audio_data, txt_data):
            audio_path = audio_line.strip().split()[1]
            audio, sr = torchaudio.load(audio_path)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            mel = feature_extractor(
                audio.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt"
            )["input_features"]
            text = " ".join(text_line.split()[1:]).lower().strip()

            item = {"mel": mel, "text": text}
            audio_dataset.append(item)

        return audio_dataset

    def evaluate(model, dataset):
        with torch.no_grad():
            all_pred, all_gt = [], []
            for item in dataset:
                mel = item["mel"]

                if torch_dtype == torch.float16:
                    mel = mel.to(torch.float16)

                generated_ids = model.generate(
                    input_features=mel.to(device),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=150,
                )
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                ## text normalization
                pred = normalizer(generated_text)
                pred = pred if len(pred) > 0 else "<UNK>"

                gt = normalizer(item["text"])
                gt = gt if len(gt) > 0 else "<UNK>"

                all_pred.append(pred)
                all_gt.append(gt)

        return calculate_wer(all_gt, all_pred)

    ## prepare dataset
    test_dataset = data_preparation(TEST_DATA, feature_extractor)
    torch.save(test_dataset, f"data/test_{DATASET}.pt")
    print(f"{DATASET}:")

    ## evaluate official whisper (only need to run once)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)

    model.eval()

    print(f"zero-shot = {evaluate(model, test_dataset)}")

    ## evaluate star adapted whisper
    model = torch.load(CKPT).to(device)
    model.eval()

    print(f"star = {evaluate(model, test_dataset)}")


if __name__ == "__main__":
    fire.Fire(inference)
