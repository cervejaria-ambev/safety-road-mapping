# General Instructions

## Generating API token

This project uses [openrouteservice API](https://openrouteservice.org) to plot maps and routes.
So the following steps are necessary at first:

1. Sign up on [openrouteservice.org](https://openrouteservice.org/dev/#/signup) to generate an API token;
2. Create a `.env` file with the following content: `TOKEN=XXXXXXXXXXXXXXX`, where `XXXXXXXXXXXXXXX` is the token generated in the step before;
3. Install the lib: `pip install safety-road-mapping`
4. Get the road accidents according the instructions bellow: [Getting accident road data](#getting-accident-road-data)
5. See the documentation on: [safety documentation](docs/safety_road_mapping/safety.md)
6. See the examples on how to use the lib on: [how-to-use.ipynb](./safety_road_mapping/examples/how-to-use.ipynb)

## Getting accident road data

- The accidents data used were extracted from the [Polícia Rodoviária Federal website](https://www.gov.br/prf/pt-br).
- The notebook [get_data.ipynb](./safety_road_mapping/extract_data/get_data.ipynb) is responsible to download and extract the data used.
- If you want to directly download the files you can [click here](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-acidentes).
- To consolidate all the accidents by year on an unique `.csv` file use the notebook [consolidate_data.ipynb](./safety_road_mapping/extract_data/consolidate_data.ipynb).
- For a simple explore data analysis from the data look at [eda_accidents.ipynb](./safety_road_mapping/extract_data/eda_accidents.ipynb)

### Adding São Paulo data

- Access the [infosiga website](http://www.respeitoavida.sp.gov.br/relatorios/)
- Download the xlsx file with fatal accidents: [Fatal accidents](http://painelderesultados.infosiga.sp.gov.br/bases/acidentes_fatais.xlsx)
- Download the csv file with non fatal accidents: [Non-fatal accidents](http://painelderesultados.infosiga.sp.gov.br/bases/acidentes_naofatais.csv)
- Use the notebook [treat_data_from_SP.ipynb](./safety_road_mapping/extract_data/treat_data_from_SP.ipynb) to transform and concatenate the data generating a final file that will be used as input for the safety map code.

## Other possible steps on the project

- The accidents data used comes just from road federal police source, so there are some routes that don't receive score because they are state highways.
- Create some unit tests.
- Use [Renaest](https://www.gov.br/infraestrutura/pt-br/assuntos/transito/arquivos-denatran/docs/renaest) data (currently only [SP has geolocation data](http://www.infosiga.sp.gov.br)).
- Include other variables in the score calculation.
- Include driver behaviors data.
- Implement an algorithm to predict accidents.
- The routes subsections are not connected, once they are plotted individually in the map. Visually it can be interesting to connect them. (Is it possible or necessary?).
