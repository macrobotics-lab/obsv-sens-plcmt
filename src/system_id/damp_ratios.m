% This script estimates damping ratios from impulse response data.
% It uses the logarithmic decrement method on the 2D tip centroid motion.

clear;
close all;

N = 2;  % Number of modes to consider (same as in corresponding params.yaml)
zeta_1s = [];  % Store zeta_1 values from each pressure trial

% Loop over trials at setpoint pressures from 0 to 100 kPa
for p_set = 0:10:100
    T = readtable(sprintf('../../data/impulse/%dkPa.csv', p_set));

    % Extract 2D position of tip centroid (Spine2D7)
    centroid7_2D = [T.Spine2D7_x';
                     T.Spine2D7_y'];
    centroid7_2D_norm = vecnorm(centroid7_2D, 2); % Norm of 2D position at each timestep

    % Compute damping ratio zeta_1 for this trial and store it
    zeta_1s = [zeta_1s, get_zeta_1(centroid7_2D_norm)];
end

% Average zeta_1 across all trials
zeta_1 = mean(zeta_1s);
disp('zeta_1:')
disp(zeta_1)

% Known natural frequencies [Hz] from FFT analysis (see nat_freqs.m)
fid = fopen('temp_params.json', 'r');
raw = fread(fid, inf, 'char=>char')';
fclose(fid);
data = jsondecode(raw);
nat_freqs = data.nat_freqs(1:N);

% Estimate beta assuming stiffness proportional damping: zeta_i = beta * omega_i / 2
beta = 2 * zeta_1 / nat_freqs(1);

% Use beta to estimate zeta_i
zetas = [zeta_1];
for i = 2:N
    zetas(i) = beta * nat_freqs(i) / 2;
    fprintf("zeta_%d:\n", i)
    disp(zetas(i))
end

% Save damping ratios to json
data.zetas = zetas;
fid = fopen('temp_params.json', 'w');
fwrite(fid, jsonencode(data), 'char');
fclose(fid);

% Compute zeta_1 using logarithmic decrement
function zeta_1 = get_zeta_1(pos_norm)
    % Find peaks in signal; invert to get peaks instead of troughs
    [pks_raw, locs_raw] = findpeaks(-pos_norm, 'MinPeakProminence', 0.1);
    
    % Filter out early peaks before external impulse
    valid_idx = locs_raw >= 200;
    pks = pks_raw(valid_idx);
    locs = locs_raw(valid_idx);
    
    % Get index of highest peak
    [~, max_idx] = max(pks);
    
    % Truncate from that point onward
    pks = pks(max_idx:end);
    locs = locs(max_idx:end);
    
    % Recover original (positive) trough values
    troughs = pos_norm(locs);  % Use the aligned locs
    
    % Shift by steady state to get peaks instead of troughs
    steadystate = mean(pos_norm(end-100:end));
    amplitudes = steadystate - troughs;
    
    % Compute logarithmic decrement
    delta = log(amplitudes(1)/amplitudes(2));
    % Compute damping ratio
    zeta_1 = delta / sqrt(delta^2 + (2*pi)^2);
end