# FedVARE: Variance-Aware Reference Selection for Federated Text Classification

Code and supplementary material for the ESANN 2027 submission
*"FedVARE: Variance-Aware Reference Selection for Federated Text Classification"*
(A. El Gharably, A. El Soukkary, N. Sakr, T. ElBatt, K. Banawan, K. Seddik — The American University in Cairo).

FedVARE is a loss-based aggregation rule for federated fine-tuning. After the first round, the server uses the
returned client models as probes, selects once the proxy samples with the highest cross-client loss variance
(a class-balanced reference set), and in every round weights each client model **and the current global model**
by their inverse loss on that set.

**[SUPPLEMENTARY.md](SUPPLEMENTARY.md)** holds the material that did not fit in the 6-page paper:
full hyperparameters for every method, Algorithm 1, the variance–range lemma with proof, the reference-set
construction details, the FedAdamW baseline details, dataset splits, and the IID results.

## Repository layout

```
notebooks/
  agnews/           T5-small + LoRA, 30 clients, Dirichlet alpha = 0.1, 50 rounds
    01_fedavg_adam.ipynb      FedAvg with Adam            (paper: "Adam")
    02_fedavg_sophia.ipynb    FedAvg with Sophia          (paper: "Sophia")
    03_fedadamw.ipynb         FedAdamW
    04_fedlbw.ipynb           FedLBW, 1,000-sample proxy (Sophia local optimizer)
    05_fedvare_adam.ipynb     FedVARE with Adam
    06_fedvare_sophia.ipynb   FedVARE with Sophia
  sentiment140/     ELECTRA-small (full fine-tuning), 100 clients (10 per round), alpha = 0.5, 100 rounds
    01 ... 06                 same six methods
  ablation/
    reference_set_selection_sentiment140.ipynb   high / mid / low variance, random, full pool (Sec. 3.2)
  generalization/
    heldout_clients_sentiment140.ipynb           ten held-out clients, stability metrics (Sec. 3.3)
```

All notebooks were run on Kaggle (2x T4 GPU). Each notebook is self-contained: data loading, Dirichlet
partitioning, local training, aggregation and evaluation.

## Running

```bash
pip install -r requirements.txt
jupyter notebook notebooks/agnews/06_fedvare_sophia.ipynb
```

- **Seeds.** Set `seed = ...` in the *Constants* cell. The paper uses five seeds:
  `342365, 8541865, 785638984, 1486496, 2658`. Within a seed, all methods share the proxy pool, client
  partition, initialization and participation schedule.
- **Data.** AG News is loaded from the Hugging Face hub (`fancyzhx/ag_news`). Sentiment140 is loaded from
  `stanfordnlp/sentiment140` (parquet); the 139 neutral test tweets are removed, leaving 359 test tweets.
  Some notebooks read a Kaggle copy of the same parquet files under `/kaggle/input/...`; point the path to
  your local copy if you run outside Kaggle.
- **Long runs.** The AG News notebooks run 25 rounds per Kaggle session and resume from the saved
  `main_model/` and `client_state/` (optimizer states) of the previous session.
- **Reference set.** In the main runs the selected reference-set indices are stored in the
  `half wagg` cell (`indx = [...]`); they are produced by the selection procedure in
  `ablation/reference_set_selection_sentiment140.ipynb` (probe round, cross-client loss variance,
  class-wise threshold with lambda = 2).

## Citation

```bibtex
@inproceedings{elgharably2027fedvare,
  title     = {{FedVARE}: Variance-Aware Reference Selection for Federated Text Classification},
  author    = {El Gharably, Aaser and El Soukkary, Ahmed and Sakr, Nouri and ElBatt, Tamer and
               Banawan, Karim and Seddik, Karim},
  booktitle = {Proc. ESANN},
  year      = {2027},
  note      = {Under review}
}
```
