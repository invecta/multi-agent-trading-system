// Chart.js Integration for Enhanced Dashboard

// Initialize Chart.js instances
let priceChart = null;
let volumeChart = null;
let performanceChart = null;

// Chart.js configuration
const chartConfig = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: true,
            position: 'top'
        },
        tooltip: {
            enabled: true,
            mode: 'index',
            intersect: false
        }
    },
    scales: {
        x: {
            display: true,
            title: {
                display: true,
                text: 'Date'
            }
        },
        y: {
            display: true,
            title: {
                display: true,
                text: 'Price'
            }
        }
    }
};

// Dark theme chart configuration
const darkChartConfig = {
    ...chartConfig,
    plugins: {
        ...chartConfig.plugins,
        legend: {
            ...chartConfig.plugins.legend,
            labels: {
                color: '#e0e0e0'
            }
        }
    },
    scales: {
        x: {
            ...chartConfig.scales.x,
            ticks: {
                color: '#e0e0e0'
            },
            grid: {
                color: '#404040'
            },
            title: {
                ...chartConfig.scales.x.title,
                color: '#e0e0e0'
            }
        },
        y: {
            ...chartConfig.scales.y,
            ticks: {
                color: '#e0e0e0'
            },
            grid: {
                color: '#404040'
            },
            title: {
                ...chartConfig.scales.y.title,
                color: '#e0e0e0'
            }
        }
    }
};

// Create price chart
function createPriceChart(data) {
    const ctx = document.getElementById('price-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (priceChart) {
        priceChart.destroy();
    }
    
    const isDarkTheme = document.body.classList.contains('dark-theme');
    const config = isDarkTheme ? darkChartConfig : chartConfig;
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels || [],
            datasets: [{
                label: 'Price',
                data: data.prices || [],
                borderColor: '#007bff',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: config
    });
}

// Create volume chart
function createVolumeChart(data) {
    const ctx = document.getElementById('volume-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (volumeChart) {
        volumeChart.destroy();
    }
    
    const isDarkTheme = document.body.classList.contains('dark-theme');
    const config = isDarkTheme ? darkChartConfig : chartConfig;
    
    volumeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels || [],
            datasets: [{
                label: 'Volume',
                data: data.volumes || [],
                backgroundColor: 'rgba(40, 167, 69, 0.6)',
                borderColor: '#28a745',
                borderWidth: 1
            }]
        },
        options: {
            ...config,
            scales: {
                ...config.scales,
                y: {
                    ...config.scales.y,
                    title: {
                        display: true,
                        text: 'Volume',
                        color: isDarkTheme ? '#e0e0e0' : '#333333'
                    }
                }
            }
        }
    });
}

// Create performance chart
function createPerformanceChart(data) {
    const ctx = document.getElementById('performance-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    const isDarkTheme = document.body.classList.contains('dark-theme');
    const config = isDarkTheme ? darkChartConfig : chartConfig;
    
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels || [],
            datasets: [{
                label: 'Portfolio Value',
                data: data.portfolioValues || [],
                borderColor: '#28a745',
                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }, {
                label: 'Benchmark',
                data: data.benchmarkValues || [],
                borderColor: '#dc3545',
                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1
            }]
        },
        options: {
            ...config,
            scales: {
                ...config.scales,
                y: {
                    ...config.scales.y,
                    title: {
                        display: true,
                        text: 'Value ($)',
                        color: isDarkTheme ? '#e0e0e0' : '#333333'
                    }
                }
            }
        }
    });
}

// Update charts when theme changes
function updateChartsTheme() {
    const isDarkTheme = document.body.classList.contains('dark-theme');
    const config = isDarkTheme ? darkChartConfig : chartConfig;
    
    if (priceChart) {
        priceChart.options = config;
        priceChart.update();
    }
    
    if (volumeChart) {
        volumeChart.options = {
            ...config,
            scales: {
                ...config.scales,
                y: {
                    ...config.scales.y,
                    title: {
                        display: true,
                        text: 'Volume',
                        color: isDarkTheme ? '#e0e0e0' : '#333333'
                    }
                }
            }
        };
        volumeChart.update();
    }
    
    if (performanceChart) {
        performanceChart.options = {
            ...config,
            scales: {
                ...config.scales,
                y: {
                    ...config.scales.y,
                    title: {
                        display: true,
                        text: 'Value ($)',
                        color: isDarkTheme ? '#e0e0e0' : '#333333'
                    }
                }
            }
        };
        performanceChart.update();
    }
}

// Listen for theme changes
document.addEventListener('DOMContentLoaded', function() {
    // Watch for theme changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                updateChartsTheme();
            }
        });
    });
    
    observer.observe(document.body, {
        attributes: true,
        attributeFilter: ['class']
    });
});

// Export functions for use in Dash callbacks
window.createPriceChart = createPriceChart;
window.createVolumeChart = createVolumeChart;
window.createPerformanceChart = createPerformanceChart;
window.updateChartsTheme = updateChartsTheme;
