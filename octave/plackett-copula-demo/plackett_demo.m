1;

% https://stackoverflow.com/questions/68908372/
% how-can-i-generate-two-weibull-random-vectors-with-a-given-correlation-coefficie


function rho = spearman_rho_log(logtheta)
    % Compute the Spearman coefficient rho from the log of the
    % theta parameter of the bivariate Plackett copula.
    %
    % The formula for rho from slide 66 of
    % http://www.columbia.edu/~rf2283/Conference/1Fundamentals%20(1)Seagers.pdf
    % rho is
    %   rho = (theta + 1)/(theta - 1) - 2*theta/(theta - 1)**2 * log(theta)
    % If R = log(theta), this can be rewritten as
    %   coth(R/2) - R/(cosh(R) - 1)
    %
    % Note, however, that the formula for the Spearman correlation rho in
    % the article "A compendium of copulas" at
    %   https://rivista-statistica.unibo.it/article/view/7202/7681
    % does not include the term log(theta).  (See Section 2.1 on the page
    % labeled 283, which is the 5th page of the PDF document.)

    rho = coth(logtheta/2) - logtheta/(cosh(logtheta) - 1);
endfunction;


function logtheta = est_logtheta(rho)
    % This function gives a pretty good estimate of log(theta) for
    % the given Spearman coefficient rho.  That is, it approximates
    % the inverse of spearman_rho_log(logtheta).

    logtheta = logit((rho + 1)/2)/0.69;
endfunction;


function theta = bivariate_plackett_theta(spearman_rho)
    % Compute the inverse of the function spearman_rho_log,
    %
    % Note that theta is returned, not log(theta).

    logtheta = fzero(@(t) spearman_rho_log(t) - spearman_rho, ...
                     est_logtheta(spearman_rho), optimset('TolX', 1e-10));
    theta = exp(logtheta);
endfunction


function [u, v] = bivariate_plackett_sample(theta, m)
    % Generate m samples from the bivariate Plackett copula.
    % theta is the parameter of the Plackett copula.
    %
    % The return arrays u and v each hold m samples from the standard
    % uniform distribution.  The samples are not independent.  The
    % expected Spearman correlation of the samples can be computed with
    % the function spearman_rho_log(log(theta)).
    %
    % The calculations are based on the information in Chapter 6 of the text
    % *Copulas and their Applications in Water Resources Engineering*
    % (Cambridge University Press).

    u = unifrnd(0, 1, [1, m]);
    w2 = unifrnd(0, 1, [1, m]);
    S = w2.*(1 - w2);
    d = sqrt(theta.*(theta + 4.*S.*u.*(1 - u).*(1 - theta).^2));
    c = 2*S.*(u.*theta.^2 + 1 - u) + theta.*(1 - 2*S);
    b = theta + S.*(theta - 1).^2;
    v = (c - (1 - 2*w2).*d)./(2*b);
endfunction

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% Main calculation
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% rho is the desired Spearman correlation.
rho = -0.75;
% m is the number of samples to generate.
m = 2000;

% theta is the Plackett copula parameter.
theta = bivariate_plackett_theta(rho);

% Generate the correlated uniform samples.
[u, v] = bivariate_plackett_sample(theta, m);

% At this point, u and v hold samples from the uniform distribution.
% u and v are not independent; the Spearman rank correlation of u and v
% should be approximately rho.
% Now use wblinv to convert u and v to samples from the Weibull distribution
% by using the inverse transform method (i.e. pass the uniform samples
% through the Weibull quantile function wblinv).
% This changes the Pearson correlation, but not the Spearman correlation.

% Weibull parameters
k = 1.6;
scale = 6.5;
wbl1 = wblinv(u, scale, k);
wbl2 = wblinv(v, scale, k);

% wbl1 and wbl2 are the correlated Weibull samples.

printf("Spearman correlation: %f\n", spearman(wbl1, wbl2))
printf("Pearson correlation:  %f\n", corr(wbl1, wbl2))

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% Plots
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% Scatter plot:

figure(1)
plot(wbl1, wbl2, '.')
axis('equal')
grid on


% Marginal histograms:

figure(2)

wbl = [wbl1; wbl2];
maxw = 1.02*max(max(wbl));
nbins = 40;

for p = 1:2
    subplot(2, 1, p)
    w = wbl(p, :);
    [nn, centers] = hist(w, nbins);
    delta = centers(2) - centers(1);
    hist(w, nbins, "facecolor", [1.0 1.0 0.7]);
    hold on
    plot(centers, delta*wblpdf(centers, scale, k)*m, 'k.')
    grid on
    xlim([0, maxw])
    if p == 1
        title("Marginal histograms")
    endif
endfor
