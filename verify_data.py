# Verify data in database
from data_collector import FinancialDataCollector

collector = FinancialDataCollector()
summary = collector.get_data_summary()

print('Database Summary:')
print(f'Total Records: {summary["total_records"]}')
print(f'Asset Classes: {summary["asset_class_counts"]}')
print(f'Timeframes: {summary["timeframe_counts"]}')
print(f'Date Range: {summary["date_range"]}')

# Test getting EUR/USD data
eurusd_data = collector.get_market_data('EURUSD=X', '1d')
print(f'\nEUR/USD Data: {len(eurusd_data)} records')
if not eurusd_data.empty:
    print(f'Date range: {eurusd_data.index[0]} to {eurusd_data.index[-1]}')
    print(f'Price range: ${eurusd_data["low"].min():.2f} - ${eurusd_data["high"].max():.2f}')
