import re

iqtree_file = snakemake.input.iqtree
epa_models = snakemake.input.epa_models

model_file = snakemake.output.model


def read_iqtree_models(iqtree_file):
    with open(iqtree_file, "r") as file:
        file_content = file.read()
    pattern = re.compile(
        r"^(\S+?)\s+[-\d.]+\s+[-\d.]+\s+[-\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+",
        re.MULTILINE,
    )
    # Find all matches
    models = pattern.findall(file_content)
    return models


substitution_matrix = [
    # DNA
    "JC",
    "K80",
    "F81",
    "HKY",
    "TN93ef",
    "TN93",
    "K81",
    "K81uf",
    "TPM2",
    "TPM2uf",
    "TPM3",
    "TPM3uf",
    # the following models have different names in epa
    # "TIM1",
    # "TIM1uf",
    # "TIM2",
    # "TIM2uf",
    # "TIM3",
    # "TIM3uf",
    # "TVMef",
    "TVM",
    "SYM",
    "GTR",
    # Amino Acids
    "Blosum62",
    "cpREV",
    "Dayhoff",
    "DCMut",
    "DEN",
    "FLU",
    "HIVb",
    "HIVw",
    "JTT",
    "JTT-DCMut",
    "LG",
    "mtART",
    "mtMAM",
    "mtREV",
    "mtZOA",
    "PMB",
    "Q.pfam",
    "Q.bird",
    "Q.insect",
    "Q.mammal",
    "Q.plant",
    "Q.yeast",
    "rtREV",
    "stmtREV",
    "VT",
    "WAG",
    "LG4M",
    "LG4X",
    "PROTGTR",
]
stationary_frequencies = [
    "F",
    "FC",
    "FU",
    # "FE",  # this is calles FQ in iqtree, we deal with this special case separately
]  # , "FU{f1/f2/../fn}", "FU{freqs.txt}"] # the last two are user-defined, which we do not do here
invariant_sites = [
    "I",
    "IO",
    "IC",
]  # , "IU{p}"] # IU{p} is user specified, which we currenly do not allow
heterogeneity_model = [
    "G",
    "G4m",
    "GA",
    # we deal with G{n} and R{n} separately below
    # "G{n}",
    # "G{n}{a}", # a is user-defined, which we do not allow here
    # "R{n}",
    # "R{n}{r1/r2/../rn}{w1/w2/../wn}", # notation for this is different in iqtree: R{n}{w1,r1,...,wn,rn}
]


iqtree_models = read_iqtree_models(iqtree_file)
for model in iqtree_models:
    components = model.split("+")
    for i in range(len(components)):
        component = components[i]
        if (
            component in substitution_matrix
            or component in stationary_frequencies
            or component in invariant_sites
            or component in heterogeneity_model
        ):
            is_epa_model = True
        # SPECIAL CASES
        # FE is called FW in epa:
        elif component == "FE":
            components[i] = "FQ"
            model = "+".join(components)
            is_epa_model = True
        elif re.match(r"TIM[1-3][^e]", component):
            component += "uf"
            model = "+".join(components)
        elif re.match(r"TIM[1-3]e", component):
            components[i] = component[:-1]
            model = "+".join(components)
        elif component == "TVMe":
            components[i] += "f"
            model = "+".join(components)
        # pattern for G{n} and R{n}
        elif re.match(r"(G|R)[0-9]+", component):
            is_epa_model = True
        else:
            is_epa_model = False
            break
    if is_epa_model:
        with open(model_file, "w") as f:
            f.write(model)
            break
