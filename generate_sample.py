import pandas as pd
import math
import argparse
from featured_data_generated import cal_pep_des
import time
import os

class GenerateSample():
    def __init__(self, grampa_path, negative_file_path, generate_example_path, mode):
        self.grampa_path = grampa_path
        self.negative_file_path = negative_file_path
        self.generate_example_path = generate_example_path
        self.mode = mode

    def __call__(self, *args, **kwargs):
        # 检查路径是否存在
        print(f"Checking paths...")
        print(f"grampa_path: {self.grampa_path}, exists: {os.path.exists(self.grampa_path)}")
        print(f"negative_file_path: {self.negative_file_path}, exists: {os.path.exists(self.negative_file_path)}")
        print(f"generate_example_path: {self.generate_example_path}, exists: {os.path.exists(self.generate_example_path)}")
        if not os.path.exists(self.generate_example_path):
            os.makedirs(self.generate_example_path)
            print(f"Created output directory: {self.generate_example_path}")

        # 读取数据
        print("Reading data...")
        data = pd.read_csv(self.grampa_path, encoding="utf8")
        print(f"grampa data loaded, shape: {data.shape}")
        data = self.filter_data_with_str("bacterium", "aureus", data)
        print(f"Filtered data, shape: {data.shape}")
        positive_sample = self.generate_all_peptides(data)
        print(f"Generated positive samples, shape: {positive_sample.shape}")
        negtive_sample = self.generate_negative_data(self.negative_file_path)
        print(f"Generated negative samples, shape: {negtive_sample.shape}")
        all_sample = self.concat_datasets(positive_sample, negtive_sample)
        print(f"Concatenated samples, shape: {all_sample.shape}")

        # 生成分类样本
        print("Generating classify sample...")
        num = len(all_sample)
        start = time.time()
        sequence = all_sample["sequence"]
        peptides = sequence.values.copy().tolist()
        result = all_sample["MIC"]
        type = all_sample["type"]
        output_path = self.generate_example_path + "classify_sample.csv"
        print(f"Classify sample output path: {output_path}")
        cal_pep_des.cal_pep(peptides, sequence, result, type, output_path)
        end = time.time()
        print("Generated classify feature data, cost time:", (end - start) / num)

        # 生成回归样本
        print("Generating regression sample...")
        if self.mode == "all":
            self.split_sample(all_sample)
        else:
            self.split_sample(positive_sample)
        print("Regression sample generation completed.")

    def filter_data_with_str(self, col_name, str, data):
        bool_filter = data[col_name].str.contains(str)
        filter_data = data[bool_filter]
        return filter_data

    def generate_all_peptides(self, data):
        data_all = [[], [], []]
        for i in data["sequence"].unique():
            data_all[0].append(i)
            log_num = 0
            count = 0
            for i in data[data["sequence"] == i]["value"]:
                log_num += math.pow(10, i)
                count += 1
            data_all[1].append(float(log_num / count))
            data_all[2].append(1)

        data_all = list(map(list, zip(*data_all)))
        data = pd.DataFrame(data=data_all, columns=["sequence", "MIC", "type"])
        return data

    def data2csv(self, data, file_name):
        print(f"Saving file to {file_name}, data shape: {data.shape}")
        data.to_csv(file_name, encoding="utf8", index=False)

    def generate_negative_data(self, negative_file_path):
        data_negative = pd.read_csv(negative_file_path, encoding="utf8")
        print(f"Negative data loaded, shape: {data_negative.shape}")
        data_negative = data_negative[~data_negative["Sequence"].str.contains("B|X|Z|O|U")]
        data_negative.reset_index(drop=True, inplace=True)
        data = pd.DataFrame(columns=["sequence", "MIC", "type"])
        rows = []
        for i in range(data_negative.shape[0]):
            rows.append({"sequence": data_negative["Sequence"][i], "MIC": 8196, "type": 0})
        data = pd.concat([data, pd.DataFrame(rows)], ignore_index=True)
        return data

    def concat_datasets(self, positive_file, negative_file):
        data_concat = pd.concat([positive_file, negative_file], ignore_index=True, axis=0)
        data_concat = data_concat.sample(frac=1, random_state=None)
        data_concat.reset_index(drop=True, inplace=True)
        return data_concat

    def split_sample(self, sample):
        num = len(sample)
        train_sample = sample[:int(0.8 * num)]
        test_sample = sample[int(0.8 * num):]
        self.data2csv(train_sample, self.generate_example_path + "regression_train_sample.csv")
        self.data2csv(test_sample, self.generate_example_path + "regression_test_sample.csv")

if __name__ == "__main__":
    grampa_path = "data/origin_data/grampa.csv"
    negative_file_path = "data/origin_data/origin_negative.csv"
    generate_example_path = "data/filtered_data/"
    mode = "all"
    generator = GenerateSample(grampa_path, negative_file_path, generate_example_path, mode)
    generator()