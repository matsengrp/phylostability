from ete3 import Tree
import pandas as pd
import pickle


def test_bts_scores():
    """
    Test bts score computation for test_input_alignment.fasta with consists of
    six species from reptile dataset of Mariadassou et al.
    Assumes that the pickled bts DataFrame for this dataset has been saved
    when running create_plots.py
    """
    true_bts_scores = [
        "Cordylus.warreni",
        "Geocalamus.acutus",
        "Lepidophyma.flavimaculatum",
        "Sceloporus.occidentalis",
        "Shinisaurus.crocodilrus",
        "Takydromus.tachydromoides",
    ]
    true_bts_scores = {
        "Sceloporys.occidentalis,Shinisaurus.crocodilurus": 75,
        "Crodylus.warreni,Lepidophyma.flavimaculatum": 50,
        "Geocalamus.acutus,Takydromus.tachydromoides": 100,
    }
    true_scores_df = pd.DataFrame(true_bts_scores, index=["bts"]).transpose()
    with open("test_data/bts_df.p", "rb") as f:
        bts_df = pickle.load(f)
    # sort columns so we can check that dfs are identical
    true_scores_df = true_scores_df.sort_values(
        by=list(true_scores_df.columns)
    ).reset_index(drop=True)
    bts_df = bts_df.sort_values(by=list(bts_df.columns)).reset_index(drop=True)

    if true_scores_df.equals(bts_df):
        print("bts_score test passed.")
    else:
        print("bts_score test failed.")


test_bts_scores()
