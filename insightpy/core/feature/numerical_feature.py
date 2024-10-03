from insightpy.core.feature import Target
from insightpy.core.feature.base_feature import Feature
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import numpy as np
class NumericalFeature(Feature):
    def summary(self):
        return {
            "mean": self.data.mean(),
            "std": self.data.std(),
            "min": self.data.min(),
            "max": self.data.max()
        }

    def detect_outliers(self, method='IQR'):
        """Stateless function that detects outliers in the feature data."""
        if method == 'IQR':
            q1 = self.data.quantile(0.25)
            q3 = self.data.quantile(0.75)
            iqr = q3 - q1
            return self.data[(self.data < (q1 - 1.5 * iqr)) | (self.data > (q3 + 1.5 * iqr))]



    def handle_numerical(self, target:Feature):
        """Handle numerical features by checking correlation, mutual information, and scaling."""
        # Check correlation
        if isinstance(target,NumericalFeature):
            corr, _ = pearsonr(self.data, target.data)
            print(f"Pearson Correlation with target: {corr}")
        else:
            corr, _ = spearmanr(self.data, target.data)
            print(f"Spearman Correlation with target: {corr}")

        # Compute mutual information for non-linear relationships
        mi = mutual_info_regression(self.data.values.reshape(-1, 1), target.data.values.reshape(-1, 1))
        print(f"Mutual Information: {mi[0]}")

        # Scaling
        scaler = StandardScaler()
        scaled_feature = scaler.fit_transform(self.data.values.reshape(-1, 1))

        # Non-linear transformation recommendation
        if corr < 0.2 and mi[0] > 0.5:
            print("Suggesting log or sqrt transformation due to non-linear relationship.")
            transformed_feature = np.log1p(self.data)  # Log transformation as an example

        return scaled_feature
