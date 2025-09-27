# Quick data collection script
from data_collector import FinancialDataCollector

collector = FinancialDataCollector()
symbols = ['GBPUSD=X', 'USDJPY=X', '^GSPC', 'GC=F', 'BTC-USD']

print('Collecting data for multiple symbols...')
for symbol in symbols:
    try:
        data = collector.collect_market_data([symbol], '1d', '3mo', 'mixed')
        print(f'{symbol}: {len(data)} records')
    except Exception as e:
        print(f'{symbol}: Error - {str(e)}')

print('Data collection completed!')

# Check what we have in the database
summary = collector.get_data_summary()
print(f'\nDatabase Summary:')
print(f'Total Records: {summary["total_records"]}')
print(f'Asset Classes: {summary["asset_class_counts"]}')
print(f'Timeframes: {summary["timeframe_counts"]}')
