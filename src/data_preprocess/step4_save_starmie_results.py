from utils import save_analysis_results

os.makedirs("analysis", exist_ok=True)
final_df = save_analysis_results(df, returnResults, file_name="analysis/tmp.csv")
final_df
