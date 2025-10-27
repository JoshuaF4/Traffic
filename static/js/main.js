// Global Variables
let currentUser = null;
let activeSignals = [];
let recentAnalyses = [];

// Utility Functions
function showAlert(message, type = 'success') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
        <span>${message}</span>
    `;
    
    const container = document.querySelector('.main-content');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        setTimeout(() => alertDiv.remove(), 5000);
    }
}

function showLoading(text = 'Processing...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loadingOverlay';
    overlay.innerHTML = `
        <div class="spinner"></div>
        <div class="loading-text">${text}</div>
    `;
    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.remove();
    }
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function setActiveNav(page) {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `/${page}`) {
            link.classList.add('active');
        }
    });
}

// Modal Functions
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
    }
}

// Close modal when clicking outside
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});

// Dashboard Functions
async function loadDashboardStats() {
    try {
        const response = await fetch('/api/dashboard-stats');
        const data = await response.json();
        
        // Update stats
        document.getElementById('totalLocations').textContent = data.stats.total_locations;
        document.getElementById('totalSignals').textContent = data.stats.total_signals;
        document.getElementById('totalAnalyses').textContent = data.stats.total_analyses;
        
        // Render active signals
        renderActiveSignals(data.active_signals);
        
        // Render recent analyses
        renderRecentAnalyses(data.recent_analyses);
        
        // Render charts
        if (data.chart_data.dates.length > 0) {
            renderEfficiencyChart(data.chart_data.dates, data.chart_data.efficiencies);
            renderVehicleChart(data.chart_data.dates, data.chart_data.vehicle_counts);
        }
    } catch (error) {
        console.error('Error loading dashboard stats:', error);
    }
}

function renderActiveSignals(signals) {
    const container = document.getElementById('activeSignalsContainer');
    if (!container) return;
    
    if (signals.length === 0) {
        container.innerHTML = `
            <div class="text-center" style="padding: 3rem; color: #64748b;">
                <i class="fas fa-traffic-light" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                <p>No active signals. Train your first signal to get started!</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = signals.map(signal => `
        <div class="card" style="margin-bottom: 1rem; animation-delay: ${Math.random() * 0.3}s;">
            <div class="flex-between">
                <div>
                    <h4 style="color: var(--dark); margin-bottom: 0.5rem;">${signal.name}</h4>
                    <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;">
                        <i class="fas fa-map-marker-alt"></i> ${signal.location}
                    </p>
                    <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <div style="font-size: 0.85rem; color: #64748b;">Efficiency</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: var(--success);">
                                ${signal.efficiency.toFixed(1)}%
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.85rem; color: #64748b;">Vehicles</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary);">
                                ${signal.vehicles}
                            </div>
                        </div>
                    </div>
                </div>
                <div>
                    <span class="badge badge-${signal.status === 'active' ? 'success' : 'danger'}">
                        ${signal.status.toUpperCase()}
                    </span>
                    <button class="btn btn-outline" style="margin-top: 1rem;" onclick="toggleSignal(${signal.id})">
                        <i class="fas fa-power-off"></i>
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

function renderRecentAnalyses(analyses) {
    const container = document.getElementById('recentAnalysesContainer');
    if (!container) return;
    
    if (analyses.length === 0) {
        container.innerHTML = `
            <div class="text-center" style="padding: 2rem; color: #64748b;">
                <p>No analyses yet</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = `
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Location</th>
                        <th>Date</th>
                        <th>Status</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    ${analyses.map(analysis => `
                        <tr>
                            <td>${analysis.location}</td>
                            <td>${formatDate(analysis.created_at)}</td>
                            <td>
                                <span class="badge badge-${analysis.status === 'completed' ? 'success' : 'warning'}">
                                    ${analysis.status}
                                </span>
                            </td>
                            <td>
                                <a href="/analysis/${analysis.id}" class="btn btn-primary" style="padding: 0.5rem 1rem; font-size: 0.9rem;">
                                    <i class="fas fa-eye"></i> View
                                </a>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

// Signal Training Functions
async function loadLocations() {
    try {
        const response = await fetch('/api/locations');
        const locations = await response.json();
        
        const container = document.getElementById('locationsContainer');
        if (!container) return;
        
        if (locations.length === 0) {
            container.innerHTML = `
                <div class="text-center" style="padding: 3rem;">
                    <i class="fas fa-map-marked-alt" style="font-size: 3rem; color: var(--primary); margin-bottom: 1rem;"></i>
                    <p style="color: #64748b; margin-bottom: 1.5rem;">No locations added yet. Create your first location to start training signals.</p>
                    <button class="btn btn-primary" onclick="openModal('addLocationModal')">
                        <i class="fas fa-plus"></i> Add First Location
                    </button>
                </div>
            `;
            return;
        }
        
        container.innerHTML = locations.map(location => `
            <div class="card">
                <div class="flex-between">
                    <div style="flex: 1;">
                        <h3 style="color: var(--dark); margin-bottom: 0.5rem;">${location.name}</h3>
                        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;">
                            <i class="fas fa-map-marker-alt"></i> ${location.address || 'No address'}
                        </p>
                        <p style="color: #64748b; font-size: 0.9rem;">
                            <i class="fas fa-road"></i> ${location.num_lanes} Lanes
                        </p>
                    </div>
                    <div>
                        <button class="btn btn-primary" onclick="startAnalysis(${location.id}, ${location.num_lanes})">
                            <i class="fas fa-play"></i> Start Analysis
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading locations:', error);
        showAlert('Failed to load locations', 'error');
    }
}

async function addLocation(formData) {
    try {
        showLoading('Creating location...');
        
        const response = await fetch('/api/location/add', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.success) {
            showAlert('Location added successfully!', 'success');
            closeModal('addLocationModal');
            loadLocations();
        } else {
            showAlert('Failed to add location', 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('Error adding location:', error);
        showAlert('An error occurred', 'error');
    }
}

function startAnalysis(locationId, numLanes) {
    openModal('analysisModal');
    
    // Set up upload areas
    const container = document.getElementById('uploadContainer');
    container.innerHTML = '';
    
    for (let i = 0; i < numLanes; i++) {
        const uploadDiv = document.createElement('div');
        uploadDiv.className = 'upload-area';
        uploadDiv.innerHTML = `
            <div class="upload-icon">
                <i class="fas fa-cloud-upload-alt"></i>
            </div>
            <h4 style="color: var(--dark); margin-bottom: 0.5rem;">Lane ${i + 1}</h4>
            <p style="color: #64748b; margin-bottom: 1rem;">Click or drag video file here</p>
            <input type="file" id="lane_${i}" accept="video/*" style="display: none;">
            <button class="btn btn-outline" onclick="document.getElementById('lane_${i}').click()">
                Choose File
            </button>
            <div id="fileName_${i}" style="margin-top: 1rem; color: var(--success); font-weight: 600;"></div>
        `;
        
        const fileInput = uploadDiv.querySelector(`#lane_${i}`);
        fileInput.addEventListener('change', (e) => {
            const fileName = e.target.files[0]?.name;
            const fileNameDiv = document.getElementById(`fileName_${i}`);
            if (fileName) {
                fileNameDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${fileName}`;
            }
        });
        
        // Drag and drop
        uploadDiv.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadDiv.classList.add('dragging');
        });
        
        uploadDiv.addEventListener('dragleave', () => {
            uploadDiv.classList.remove('dragging');
        });
        
        uploadDiv.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadDiv.classList.remove('dragging');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('video/')) {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;
                document.getElementById(`fileName_${i}`).innerHTML = `<i class="fas fa-check-circle"></i> ${file.name}`;
            }
        });
        
        container.appendChild(uploadDiv);
    }
    
    // Set up submit button
    document.getElementById('submitAnalysis').onclick = () => submitAnalysis(locationId, numLanes);
}

async function submitAnalysis(locationId, numLanes) {
    const formData = new FormData();
    
    // Check all files are selected
    let allFilesSelected = true;
    for (let i = 0; i < numLanes; i++) {
        const fileInput = document.getElementById(`lane_${i}`);
        if (!fileInput.files || fileInput.files.length === 0) {
            allFilesSelected = false;
            break;
        }
        formData.append(`lane_${i}`, fileInput.files[0]);
    }
    
    if (!allFilesSelected) {
        showAlert(`Please select video files for all ${numLanes} lanes`, 'error');
        return;
    }
    
    closeModal('analysisModal');
    showLoading('Processing videos... This may take several minutes.');
    
    try {
        const response = await fetch(`/api/location/${locationId}/analysis`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Poll for analysis completion
            pollAnalysisStatus(data.analysis_id);
        } else {
            hideLoading();
            showAlert('Failed to start analysis', 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('Error submitting analysis:', error);
        showAlert('An error occurred', 'error');
    }
}

async function pollAnalysisStatus(analysisId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/analysis/${analysisId}/status`);
            const data = await response.json();
            
            if (data.completed) {
                clearInterval(pollInterval);
                hideLoading();
                showAlert('Analysis completed successfully!', 'success');
                setTimeout(() => {
                    window.location.href = `/analysis/${analysisId}`;
                }, 1500);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                hideLoading();
                showAlert('Analysis failed. Please try again.', 'error');
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 3000); // Poll every 3 seconds
}

// Signal Inventory Functions
async function loadSignals() {
    try {
        const response = await fetch('/api/signals');
        const signals = await response.json();
        
        const container = document.getElementById('signalsContainer');
        if (!container) return;
        
        if (signals.length === 0) {
            container.innerHTML = `
                <div class="text-center" style="padding: 3rem;">
                    <i class="fas fa-traffic-light" style="font-size: 3rem; color: var(--primary); margin-bottom: 1rem;"></i>
                    <p style="color: #64748b;">No signals in inventory yet.</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = `
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Signal Name</th>
                            <th>Location</th>
                            <th>Status</th>
                            <th>Efficiency</th>
                            <th>Vehicles Processed</th>
                            <th>Last Active</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${signals.map(signal => `
                            <tr>
                                <td><strong>${signal.name}</strong></td>
                                <td>${signal.location}</td>
                                <td>
                                    <span class="badge badge-${signal.status === 'active' ? 'success' : signal.status === 'inactive' ? 'danger' : 'warning'}">
                                        ${signal.status}
                                    </span>
                                </td>
                                <td>
                                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                                        <div style="flex: 1; background: #e2e8f0; border-radius: 10px; height: 8px; overflow: hidden;">
                                            <div style="width: ${signal.efficiency_score}%; background: var(--success); height: 100%;"></div>
                                        </div>
                                        <span style="font-weight: 600;">${signal.efficiency_score.toFixed(1)}%</span>
                                    </div>
                                </td>
                                <td>${signal.vehicles_processed.toLocaleString()}</td>
                                <td>${formatDate(signal.last_active)}</td>
                                <td>
                                    <div style="display: flex; gap: 0.5rem;">
                                        <button class="btn btn-${signal.status === 'active' ? 'danger' : 'success'}" 
                                                style="padding: 0.5rem 1rem; font-size: 0.9rem;"
                                                onclick="toggleSignal(${signal.id})">
                                            <i class="fas fa-power-off"></i>
                                        </button>
                                        <a href="/signal-training" class="btn btn-primary" style="padding: 0.5rem 1rem; font-size: 0.9rem;">
                                            <i class="fas fa-edit"></i>
                                        </a>
                                    </div>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    } catch (error) {
        console.error('Error loading signals:', error);
        showAlert('Failed to load signals', 'error');
    }
}

async function toggleSignal(signalId) {
    try {
        const response = await fetch(`/api/signal/${signalId}/toggle`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(`Signal ${data.status === 'active' ? 'activated' : 'deactivated'} successfully`, 'success');
            
            // Reload appropriate data
            if (document.getElementById('signalsContainer')) {
                loadSignals();
            }
            if (document.getElementById('activeSignalsContainer')) {
                loadDashboardStats();
            }
        } else {
            showAlert('Failed to toggle signal', 'error');
        }
    } catch (error) {
        console.error('Error toggling signal:', error);
        showAlert('An error occurred', 'error');
    }
}

// Chart Functions
function renderEfficiencyChart(dates, efficiencies) {
    const canvas = document.getElementById('efficiencyChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    if (dates.length === 0) return;
    
    // Calculate dimensions
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;
    
    // Draw axes
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    
    // Draw grid lines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = padding + (chartHeight / 5) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
    }
    
    // Draw line
    const maxValue = Math.max(...efficiencies, 100);
    const stepX = chartWidth / (dates.length - 1);
    
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    efficiencies.forEach((value, index) => {
        const x = padding + stepX * index;
        const y = height - padding - (value / maxValue) * chartHeight;
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        
        // Draw point
        ctx.fillStyle = '#6366f1';
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
    });
    
    ctx.stroke();
    
    // Draw labels
    ctx.fillStyle = '#64748b';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    
    dates.forEach((date, index) => {
        const x = padding + stepX * index;
        ctx.fillText(date, x, height - padding + 20);
    });
}

function renderVehicleChart(dates, counts) {
    const canvas = document.getElementById('vehicleChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    if (dates.length === 0) return;
    
    // Calculate dimensions
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;
    
    // Draw axes
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    
    // Draw bars
    const maxValue = Math.max(...counts);
    const barWidth = chartWidth / dates.length * 0.8;
    const gap = chartWidth / dates.length * 0.2;
    
    counts.forEach((value, index) => {
        const barHeight = (value / maxValue) * chartHeight;
        const x = padding + (barWidth + gap) * index + gap / 2;
        const y = height - padding - barHeight;
        
        // Gradient fill
        const gradient = ctx.createLinearGradient(x, y, x, height - padding);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(1, '#8b5cf6');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth, barHeight);
        
        // Value on top
        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(value.toString(), x + barWidth / 2, y - 5);
    });
    
    // Draw labels
    ctx.fillStyle = '#64748b';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    
    dates.forEach((date, index) => {
        const x = padding + (barWidth + gap) * index + gap / 2 + barWidth / 2;
        ctx.fillText(date, x, height - padding + 20);
    });
}

// Form Handlers
function handleLocationForm(e) {
    e.preventDefault();
    
    const formData = {
        name: document.getElementById('locationName').value,
        address: document.getElementById('locationAddress').value,
        city: document.getElementById('locationCity').value,
        latitude: document.getElementById('locationLat').value,
        longitude: document.getElementById('locationLng').value,
        num_lanes: document.getElementById('numLanes').value
    };
    
    addLocation(formData);
}

// Authentication Functions
async function handleLogin(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    showLoading('Logging in...');
    
    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.success) {
            window.location.href = '/home';
        } else {
            showAlert('Invalid credentials', 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('Login error:', error);
        showAlert('An error occurred', 'error');
    }
}

async function handleRegister(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    
    if (password !== confirmPassword) {
        showAlert('Passwords do not match', 'error');
        return;
    }
    
    showLoading('Creating account...');
    
    try {
        const response = await fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `username=${encodeURIComponent(username)}&email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.success) {
            showAlert('Registration successful! Redirecting to login...', 'success');
            setTimeout(() => {
                window.location.href = '/login';
            }, 2000);
        } else {
            showAlert(data.error || 'Registration failed', 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('Registration error:', error);
        showAlert('An error occurred', 'error');
    }
}

// Initialize tooltips
function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(element => {
        element.addEventListener('mouseenter', (e) => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = e.target.getAttribute('data-tooltip');
            tooltip.style.position = 'absolute';
            tooltip.style.background = '#1e293b';
            tooltip.style.color = 'white';
            tooltip.style.padding = '0.5rem 1rem';
            tooltip.style.borderRadius = '8px';
            tooltip.style.fontSize = '0.85rem';
            tooltip.style.zIndex = '9999';
            
            document.body.appendChild(tooltip);
            
            const rect = e.target.getBoundingClientRect();
            tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
            tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
            
            element._tooltip = tooltip;
        });
        
        element.addEventListener('mouseleave', (e) => {
            if (e.target._tooltip) {
                e.target._tooltip.remove();
            }
        });
    });
}

// Real-time updates simulation
function simulateRealTimeUpdates() {
    setInterval(() => {
        // Update last active times
        document.querySelectorAll('[data-last-active]').forEach(element => {
            const timestamp = element.getAttribute('data-last-active');
            element.textContent = formatDate(timestamp);
        });
    }, 60000); // Update every minute
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltips
    initTooltips();
    
    // Set up real-time updates
    simulateRealTimeUpdates();
    
    // Page-specific initializations
    const currentPath = window.location.pathname;
    
    if (currentPath === '/home') {
        loadDashboardStats();
        setActiveNav('home');
    } else if (currentPath === '/signal-training') {
        loadLocations();
        setActiveNav('signal-training');
    } else if (currentPath === '/signal-inventory') {
        loadSignals();
        setActiveNav('signal-inventory');
    }
});

// Export functions for global use
window.trafficMonitor = {
    showAlert,
    showLoading,
    hideLoading,
    openModal,
    closeModal,
    loadDashboardStats,
    loadLocations,
    loadSignals,
    toggleSignal,
    handleLocationForm,
    handleLogin,
    handleRegister,
    startAnalysis
};