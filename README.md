# General Instructions

## Generating API token

This project uses [openrouteservice API](https://openrouteservice.org) to plot maps and routes.
So the following steps are necessary at first:

1. Sign up on [openrouteservice.org](https://openrouteservice.org/dev/#/signup) to generate an API token;
2. Rename the `.env.example` file inside `safety_road_mapping/` folder to `.env`;
3. Insert the generated token in the front of `TOKEN=`.

## Accident road data

The accidents data used were extracted from the Polícia Rodoviária Federal website.
The notebook `get_data.ipynb` inside `safety_road_mapping/notebooks` folder is responsible to download and extract the data used. If you want to directly download the files you can [click here](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-acidentes).
