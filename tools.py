import yfinance as yf
import pandas as pd
import random
import h2o

# Initialize H2O if not already running
h2o.init()

# Load the saved model
loaded_model = h2o.load_model('GBM_grid_1_AutoML_2_20240903_184346_model_1')

def get_ticker_data(ticker):
    financial_details_list = []
    average_debt_by_ebitda = 3.0735

    # Fetch the company data using yfinance
    company = yf.Ticker(ticker)

    # Fetch the balance sheet, income statement, cash flow, and info
    balance_sheet = company.balance_sheet
    income_statement = company.financials
    cash_flow = company.cashflow
    info = company.info

    # Extract key financial data with error handling
    total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
    total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
    total_equity = total_assets - total_liabilities if total_assets is not None and total_liabilities is not None else None
    total_revenue = income_statement.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_statement.index else None
    net_income = income_statement.loc['Net Income'].iloc[0] if 'Net Income' in income_statement.index else None
    current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else None
    current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else None
    ebit = income_statement.loc['EBIT'].iloc[0] if 'EBIT' in income_statement.index else None
    interest_expense = income_statement.loc['Interest Expense'].iloc[0] if 'Interest Expense' in income_statement.index else None
    ebitda = income_statement.loc['EBITDA'].iloc[0] if 'EBITDA' in income_statement.index else None
    total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else None

    # Calculate financial ratios with more explicit checks for NoneType
    debt_to_equity_ratio = (total_liabilities / total_equity) if total_liabilities is not None and total_equity not in (None, 0) else None
    current_ratio = (current_assets / current_liabilities) if current_assets is not None and current_liabilities not in (None, 0) else None
    roa = (net_income / total_assets) if net_income is not None and total_assets not in (None, 0) else None
    roe = (net_income / total_equity) if net_income is not None and total_equity not in (None, 0) else None
    profit_margin = (net_income / total_revenue) if net_income is not None and total_revenue not in (None, 0) else None
    interest_coverage_ratio = (ebit / interest_expense) if ebit is not None and interest_expense not in (None, 0) else None
    debt_by_ebitda = total_debt/ebitda if total_debt is not None and ebitda not in (None, 0) else None

    max_debt = average_debt_by_ebitda * ebitda
    if max_debt > total_debt:
      debtlimit = max_debt - total_debt
    else:
      debtlimit = 0

    # data that will pass through the model built below:
    ticker_data = {
      'Total Assets': [total_assets],
      'Total Liabilities': [total_liabilities],
      'Total Equity': [total_equity],
      'Total Debt': [total_debt],
      'Total Revenue': [total_revenue],
      'Net Income': [net_income],
      'EBITDA': [ebitda],
      'Debt to Equity Ratio': [debt_to_equity_ratio],
      'Current Ratio': [current_ratio],
      'ROA': [roa],
      'ROE': [roe],
      'Profit Margin': [profit_margin],
      'Interest Coverage Ratio': [interest_coverage_ratio],
      'Debt to EBITDA Ratio': [debt_to_equity_ratio],
      'Market Cap':[info.get('marketCap', 'N/A')],
      'Max Debt': [max_debt],
      'Debt Limit': [debtlimit]
    }

    return ticker_data

def load_and_predict(new_data_df):
    global loaded_model

    # Convert new data to H2OFrame if needed
    h2o_new_data_df = h2o.H2OFrame(new_data_df)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(h2o_new_data_df)
    print("\nPredictions using the loaded model:")
    print(predictions)
    return predictions

def predict_for_ticker(ticker):
    ticker_data = get_ticker_data(ticker)
    ticker_data_df = pd.DataFrame(ticker_data)
    prediction = load_and_predict(ticker_data_df)
    return prediction
