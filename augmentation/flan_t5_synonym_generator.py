import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import click
import logging
#from utils.utils import device_cuda_mps_cpu

def device_cuda_cpu():
    # seting up the device cuda, mps or cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    return device

@click.command()
@click.option(
    "--model_name",
    help="select the model to be used",
    required=False,
    default="google/flan-t5-base",
)
@click.option(
    "--prompt",
    help="select the prompt to be used",
    required=False,
    #default="dá-me 5 sinonimos novos que não constam nesta lista de palavras: hipertensão, hipertensão arterial, pressão alta, tensão alta, tensão arterial elevada",
    #default="dá-me 5 sinonimos para hipertensão arterial",
    default="give me a synonym for high blood pressure"
)
def flan_t5_synonym_generator(
    model_name="google/flan-t5-base",
    #prompt="dá-me 5 sinónimos novos que não constam nesta lista de palavras: hipertensão, hipertensão arterial, pressão alta, tensão alta, tensão arterial elevada",
    #prompt="dá-me 5 sinonimos para hipertensão arterial",
    prompt="give me a synonym for high blood pressure"
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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

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
    print(pipe(prompt))

    return pipe(prompt)


if __name__ == "__main__":
    flan_t5_synonym_generator()
