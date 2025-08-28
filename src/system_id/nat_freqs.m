% This script computes the dominant natural frequencies of the system
% from multiple impulse response trials at different pressure setpoints.

clear;
close all;

all_locs = {};  % Store dominant frequency peaks from each trial

% Loop over trials at setpoint pressures from 0 to 100 kPa
for p_set = 0:10:100
    T = readtable(sprintf('../../data/impulse/%dkPa.csv', p_set));

    % Extract 3D position of tip centroid (Spine7)
    centroid7_pos = [T.Spine7_x';
                     T.Spine7_y';
                     T.Spine7_z'];
    centroid7_pos_norm = vecnorm(centroid7_pos); % Norm of 3D position at each timestep

    % Extract dominant frequencies for this trial
    all_locs{end + 1} = get_locs(centroid7_pos_norm);
end

% Parameters for clustering frequency peaks across trials
proximity_threshold = 1;  % [Hz] — cluster if peaks are within 1 Hz of each other
min_count = 5;  % Minimum number of trials a peak must appear in

% Flatten all peaks into a single sorted list
loc_list = [];
for i = 1:length(all_locs)
    loc_list = [loc_list, all_locs{i}];
end
locs = sort(loc_list);

% Cluster peaks by proximity
clusters = {};
current_cluster = locs(1);

for i = 2:length(locs)
    % Add to current cluster if within threshold of any element
    if any(abs(locs(i) - current_cluster) <= proximity_threshold)
        current_cluster(end + 1) = locs(i);
    else
        % Save current cluster if it meets count threshold
        if numel(current_cluster) >= min_count
            clusters{end + 1} = current_cluster;
        end
        % Start new cluster
        current_cluster = locs(i);
    end
end

% Check last cluster
if numel(current_cluster) >= min_count
    clusters{end+1} = current_cluster;
end

% Compute mean frequency of each cluster
cluster_means = cellfun(@mean, clusters);

disp('Natural frequencies [Hz]:')
disp(cluster_means)

disp('Natural frequencies [rad/s]:')
disp(cluster_means * 2 * pi)

data.nat_freqs = cluster_means * 2 * pi;
% Save to json
fid = fopen('temp_params.json', 'w');
fwrite(fid, jsonencode(data), 'char');
fclose(fid);

function freq_locs = get_locs(pos_norm)
    % get_locs computes the dominant frequencies from the norm of a position vector
    % using FFT and peak detection on a smoothed power spectral density

    dt = 0.01;  % Time step [s] — 100 Hz sampling
    n = length(pos_norm);

    fhat = fft(pos_norm, n);  % Compute FFT
    PSD = fhat.*conj(fhat)/n;  % Compute PSD

    freq = 1/(dt*n)*(0:n);  % Frequency axis [Hz]
    L = 1:floor(n/2);  % Only positive frequencies up to Nyquist
    
    % Smooth the PSD (in dB) to reduce noise and make peaks more noticeable
    smoothed_PSD_dB = movmean(20 * log10(PSD(L)), 10);

    % Find peaks in smoothed PSD
    [~, locs] = findpeaks(smoothed_PSD_dB, 'MinPeakProminence', 25);
    
    % Map peak indices to corresponding frequency values
    freqs_to_Nyquist = freq(L);
    freq_locs = freqs_to_Nyquist(locs);
end
