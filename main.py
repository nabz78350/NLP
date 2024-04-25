import pandas as pd
from models import *
from modelling import *
from tqdm import tqdm
from utils import *


def main():
    models = ["bert", "mlp", "knn", "xgb", "nb", "logreg", "benchmark"]
    models = ["benchmark"]
    customs = [True, False]
    misspelling_method = ["enhanced", "lstm", "fuzzy", "none"]

    all_results = []
    for custom in customs:
        for method in misspelling_method:
            data_class = DataClass(
                use_prediction=True,
                use_enhanced=method,
                custom=custom,
                augmented_data=False,
                augmented_size=2000,
            )
            data_class.create_dataset()
            modelling_data = DataModel(
                data=data_class,
                use_enhanced=data_class.use_enhanced,
                custom_test_index=data_class.enhanced_index,
                max_len=100,
                custom=data_class.custom,
            )
            modelling_data.create_padding()
            for model in tqdm(models):
                model_socface = Model(data=modelling_data, model_name=model)
                model_socface.fit()
                model_socface.predict()
                model_socface.compute_results()
                print(model_socface.results)
                all_results.append(model_socface.results)

    output_dir = "results"
    check_or_create_directory(output_dir)

    results = pd.concat(all_results, axis=0)
    results.to_excel(os.path.join(output_dir, "results.xlsx"))


if __name__ == "__main__":
    main()
