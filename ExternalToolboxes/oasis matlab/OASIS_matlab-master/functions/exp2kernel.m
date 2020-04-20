function kernel = exp2kernel(taus, sampling, max_time)
% by Mattia, 01.20
%% create the convolution kernel (difference of two exponential) given tau_rise and tau_decay 

%% inputs: 
%   taus:     1*2 vector, [tau_decay tau_rise] - in ms! note that the order 
%             does not actually matter, because of the denominator
%   sampling: sampling rate of the vector to convolve (in Hz)
%   max_time:     scalar, length of the kernel in ms * sampling 

%% outputs: 
%   kernel:     normalized convolution kernel in steps that are equal to
%               the sampling rate

time = (1 : max_time)';
taus = taus / sampling;
decay_exp = - 1 / taus(1); % set the exponent for decay (or rise, for that matter)
rise_exp =  - 1 / taus(2); % set the exponent for rise (or decay, for that matter)
% compute the kernel
kernel = (exp(decay_exp * time) - exp(rise_exp * time)) / (exp(decay_exp) - exp(rise_exp));
% normalize the kernel
kernel = kernel / sum(kernel);

end
