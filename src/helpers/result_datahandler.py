import json
import pandas as pd
import numpy as np
from typing import Tuple

class ResultsDataHandler:
    def __init__(self, path: str) -> None:
        with open(path) as f:
            self.raw_data = json.load(f)
        self.df, self.hyperparam_df = self._process_data()

    def _process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.raw_data
        rows = []

        for fun, alphas in data.items():
            for alpha, results in alphas.items():
                penalty = results.get('best_lambda', np.nan)  # Default to NaN if 'best_lambda' is not available

                row = {
                    'Function': fun, 
                    'Alpha': alpha, 
                    'Best_Temperature': results['best_temperature'],
                    'Coverage_Rate': results['test_results']['Coverage_rate'],
                    'Average_Size': results['test_results']['Average_size'],
                    'Unilable_share': results['test_results']['Unilable_share'],
                    'Brier_score': results['test_results']['Multiclass_brier_score'],
                    'Log_loss': results['test_results']['Log_loss']
                    
                }
                
                if fun in ['RAPS', 'SAPS']:
                    row['Lambda'] = results.get('best_lambda', np.nan)
                
                rows.append(row)

        processed_data = pd.DataFrame(rows)

        metrics = ['Coverage_Rate','Brier_score', 'Log_loss', 'Average_Size', 'Unilable_share']
        hyperparams = ['Function', 'Alpha', 'Best_Temperature', 'Lambda']
        
        # Extract hyperparameters
        hyperparam_df = processed_data[hyperparams]

        # Pivot the DataFrame to get a suitable format
        df = processed_data.pivot_table(index='Alpha', columns='Function', values=metrics).reindex(columns=metrics, level=0)#.round(3)

        return df, hyperparam_df

    def get_styled_dataframe(self):
        """
        Get the styled DataFrame with highlights for average size and coverage rate.
        
        Returns:
            Tuple[pd.io.formats.style.Styler, pd.io.formats.style.Styler]: The styled DataFrames.
        """
        def highlight_min_avg_size(s):
            is_min = s == s.min()
            return ['background-color: coral' if v else '' for v in is_min]

        def highlight_max_coverage(s):
            is_max = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_max]

        # Extract the 'Coverage_Rate' and 'Average_Size' from the pivoted DataFrame
        avg_size_df = pd.DataFrame(self.df['Average_Size'])
        coverage_df = pd.DataFrame(self.df['Coverage_Rate'])

        # Apply styling separately
        styled_avg_size_df = avg_size_df.style.apply(highlight_min_avg_size, axis=1)
        styled_coverage_df = pd.DataFrame(coverage_df).style.apply(highlight_max_coverage, axis=1)

        return styled_avg_size_df, styled_coverage_df

    def save_hyperparam_df(self, file_path: str) -> None:
        """
        Save the hyperparameter DataFrame to a CSV file.
        
        Args:
            file_path (str): The path to the CSV file.
        """
        self.hyperparam_df.to_csv(file_path, index=False)
        print(f"Hyperparameter DataFrame saved to {file_path}")

if __name__ =='__main__':
    res = ResultsDataHandler('results_with_temperature.json')
    
    # Get styled DataFrames
    styled_avg_size_df, styled_coverage_df = res.get_styled_dataframe()

    # Display the styled DataFrames
    print(styled_avg_size_df)
    print(styled_coverage_df)

    # Save hyperparameter DataFrame to CSV
    res.save_hyperparam_df('hyperparams.csv')
