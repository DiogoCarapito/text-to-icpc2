import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import click
import logging

# from utils.utils import device_cuda_mps_cpu


def device_cuda_cpu():
    # seting up the device cuda, mps or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


@click.command()
@click.option(
    "--model_name",
    help="select the model to be used",
    required=False,
    default="mistralai/Mistral-7B-Instruct-v0.3",
)
@click.option(
    "--prompt",
    help="select the prompt to be used",
    required=False,
    # default="dá-me 5 sinonimos novos que não constam nesta lista de palavras: hipertensão, hipertensão arterial, pressão alta, tensão alta, tensão arterial elevada",
    # default="dá-me 5 sinonimos para hipertensão arterial",
    default="give me a synonym for high blood pressure",
)
def mistral_synonym_generator(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    prompt="dá-me 5 sinónimos novos que não constam nesta lista de palavras: hipertensão, hipertensão arterial, pressão alta, tensão alta, tensão arterial elevada",
    # prompt="dá-me 5 sinonimos para hipertensão arterial",
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # device check
    device = device_cuda_cpu()
    logging.info("Using the device '%s'", device)

    # load the model
    logging.info("Loading the pipeline %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # load the tokenizer
    logging.info("Loading the tokenizer %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the pipeline with the model and tokenizer
    logging.info("Creating the pipeline")
    pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, device=device
    )

    # generate the synonyms
    logging.info("Generating the synonyms")
    output = pipe(prompt)
    
    print(output)

    return output


if __name__ == "__main__":
    mistral_synonym_generator()
