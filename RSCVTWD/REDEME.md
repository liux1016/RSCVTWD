Regarding this project, the environment for initiation is as follows:
version:
Python interpreter 3.12
pandas 2.2.2
scikit-learn 1.4.2

The file RSCVTWD.py is a simple implementation of the proposed algorithm in this paper, where the division of equivalent classes is the simplest based on the value of the feature. You can modify it to other division methods. You can also set thresholds for different decision equivalent classes, and the algorithm in this paper uses a fixed threshold.
RSCVTWD.py and SCVTWD.py can be configured (marked with TODO):
data = pd.read_csv('/path/to/your/real_data.csv')
K = 7             ps. number of cross-validations
a = 0.7           ps. threshold α
b = 0.3           ps. threshold β