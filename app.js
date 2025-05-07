// Healthcare Fraud Detection App JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize AOS animations with settings that don't interfere with form inputs
    if (typeof AOS !== 'undefined') {
        AOS.init({
            disable: window.innerWidth < 768, // Disable on mobile
            once: true,
            disableMutationObserver: false,
            duration: 800,
            startEvent: 'load' // Initialize on window load instead of DOMContentLoaded
        });
    }
    
    // For forms in AOS elements, add special handling
    document.querySelectorAll('[data-aos] form').forEach(function(form) {
        // Remove AOS attributes from the form to prevent interference
        form.setAttribute('data-aos', 'none');
        
        // Add event listeners to all form inputs
        form.querySelectorAll('input, select, textarea, button').forEach(function(input) {
            // Enable pointer events explicitly
            input.style.pointerEvents = 'auto';
            
            // Clear any AOS attributes
            input.removeAttribute('data-aos');
            input.removeAttribute('data-aos-delay');
            input.removeAttribute('data-aos-duration');
            
            // Stop event propagation to prevent AOS from intercepting
            ['focus', 'click', 'keydown', 'input'].forEach(function(eventType) {
                input.addEventListener(eventType, function(e) {
                    e.stopPropagation();
                });
            });
        });
    });

    // Form validation styling enhancement
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });

        // Add sample data fill functionality
        const fillSampleBtn = document.getElementById('fillSampleBtn');
        if (fillSampleBtn) {
            fillSampleBtn.addEventListener('click', function() {
                // Sample data for testing
                const sampleData = {
                    'renaldiseaseindicator': '1',
                    'dr_age': '65',
                    'dr_inscclaimamtreimbursed': '6084.00',
                    'dr_ipannualreimbursementamt': '12000.00',
                    'dr_opannualreimbursementamt': '5000.00',
                    'dr_deductibleamtpaid': '500.00',
                    'dr_ipannualdeductibleamt': '1000.00',
                    'dr_opannualdeductibleamt': '500.00',
                    'dr_los': '5.0',
                    'dr_duration': '14.0'
                };
                
                // Fill the form with sample data
                Object.keys(sampleData).forEach(function(key) {
                    const input = document.getElementById(key);
                    if (input) {
                        input.value = sampleData[key];
                    }
                });
                
                // Add visual feedback
                fillSampleBtn.classList.add('btn-success');
                fillSampleBtn.innerHTML = '<i class="bx bx-check me-2"></i>Sample Data Filled';
                
                setTimeout(function() {
                    fillSampleBtn.classList.remove('btn-success');
                    fillSampleBtn.innerHTML = '<i class="bx bx-data me-2"></i>Fill Sample Data';
                }, 2000);
            });
        }

        // Allow form inputs to be editable
        const formInputs = form.querySelectorAll('input, select');
        formInputs.forEach(input => {
            // Make sure input events propagate correctly
            input.addEventListener('focus', function(e) {
                e.stopPropagation();
            });
            
            input.addEventListener('click', function(e) {
                e.stopPropagation();
            });
            
            // For number inputs, ensure they can be incremented/decremented
            if (input.type === 'number') {
                input.addEventListener('keydown', function(e) {
                    e.stopPropagation();
                });
            }
        });
    }

    // Feature importance chart
    const featureImportanceEl = document.getElementById('featureImportanceChart');
    if (featureImportanceEl && typeof Chart !== 'undefined' && typeof featureImportance !== 'undefined') {
        const ctx = featureImportanceEl.getContext('2d');
        
        // Sort feature importance for better visualization
        const sortedEntries = Object.entries(featureImportance).sort((a, b) => b[1] - a[1]);
        const labels = sortedEntries.map(entry => {
            // Make feature names more readable
            const name = entry[0];
            return name.replace('dr_', '').replace(/([A-Z])/g, ' $1')
              .replace(/^./, str => str.toUpperCase())
              .replace('amt', 'Amount')
              .replace('ip', 'Inpatient')
              .replace('op', 'Outpatient')
              .replace('los', 'Length of Stay')
              .replace('inscclaimamtreimbursed', 'Claim Amount Reimbursed');
        });
        const values = sortedEntries.map(entry => entry[1]);
        
        const featureChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Importance',
                    data: values,
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1,
                    borderRadius: 5
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Feature'
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.9)',
                        titleFont: {
                            size: 14
                        },
                        bodyFont: {
                            size: 13
                        },
                        padding: 10,
                        caretSize: 8,
                        cornerRadius: 6,
                        displayColors: false
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeOutQuart'
                },
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    // Probability chart on results page
    const probabilityChartEl = document.getElementById('probabilityChart');
    if (probabilityChartEl && typeof Chart !== 'undefined' && typeof probability !== 'undefined') {
        const ctx = probabilityChartEl.getContext('2d');
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: [fraudLabel === 1 ? 'Fraud' : 'Not Fraud', fraudLabel === 1 ? 'Not Fraud' : 'Fraud'],
                datasets: [{
                    data: [probability, 1 - probability],
                    backgroundColor: [
                        fraudLabel === 1 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(34, 197, 94, 0.8)', 
                        fraudLabel === 1 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)'
                    ],
                    borderColor: [
                        fraudLabel === 1 ? 'rgba(239, 68, 68, 1)' : 'rgba(34, 197, 94, 1)', 
                        fraudLabel === 1 ? 'rgba(239, 68, 68, 0.5)' : 'rgba(34, 197, 94, 0.5)'
                    ],
                    borderWidth: 1,
                    cutout: '70%',
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            font: {
                                size: 14
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.9)',
                        titleFont: {
                            size: 14
                        },
                        bodyFont: {
                            size: 13
                        },
                        padding: 10,
                        caretSize: 8,
                        cornerRadius: 6,
                        displayColors: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                let value = context.raw || 0;
                                return label + ': ' + (value * 100).toFixed(2) + '%';
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    animateScale: true,
                    duration: 2000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }
}); 