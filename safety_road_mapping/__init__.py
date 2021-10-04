def get_version():
    import re
    toml_file = open('../pyproject.toml').read()
    return re.findall(r'(?<=version = ").*(?="\n)', toml_file)[0]


__version__ = get_version()
