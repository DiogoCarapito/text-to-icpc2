import os
import dotenv
import click
from openai import OpenAI


@click.command()
@click.option(
    "--descricao",
    help="Descrição da doença ou problema de saúde",
    default="hipertensão arterial",
    required=False,
)
def main(descricao="hipertensão arterial"):
    # load the environment variables
    dotenv.load_dotenv()

    # get the openai client with api key and project id
    client = OpenAI(
        project=os.getenv("OPENAI_PROJECT_ID"), api_key=os.getenv("OPENAI_API_KEY")
    )

    # send the request to the openai api
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "O teu objetivo é encontrar 5 sinonimos de para a seguinte doença ou problema de saúde em português de Portugal. Os 5 sinónimos deverão ser entregues numa lista de strings em python.",
            },
            {
                "role": "user",
                "content": descricao,
            },
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1,
    )

    # print the full api response
    print(response)
    print("---")
    # get the main content of the response
    print(response.choices[0].message.content)

    return response.choices[0].message.content


if __name__ == "__main__":
    main()
