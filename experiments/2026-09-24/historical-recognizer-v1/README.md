# Historical recognizer v1

Frozen training pilot for historical Polish print. The experiment specializes
the pinned mixed-v3 recognizer without using the geometry holdout collections or
the final test collections.

- [Protocol and promotion gates](../../../docs/HISTORICAL_RECOGNIZER_V1.md)
- [Google Colab notebook](../../../training/colab_historical_recognizer_v1.ipynb)
- `config.json`: pinned inputs and hyperparameters.
- `corpus-summary.json`: aggregate result of the local deterministic corpus
  build. It contains no scans, line crops or label text.

The model has not been trained yet. Run the generated Colab notebook and return
its evidence ZIP before considering publication or test-set evaluation.
