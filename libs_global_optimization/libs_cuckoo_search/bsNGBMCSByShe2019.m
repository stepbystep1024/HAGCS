function [x,fval,exitFlag,output] = bsNGBMCSByShe2019(objFunc, Lb, Ub, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% This code modifies the Modified Cuckoo Search (MCS) 
% algorithm implemented by Walton
%
% Programmed by Bin She (bin.stepbystep@gmail.com)
% Programmed dates: May 2019
%
% -------------------------------------------------------------------------
% The work is based on the following paper:
%
% 1) %S.Walton, O.Hassan, K.Morgan and M.R.Brown "Modified cuckoo search: A
% new gradient free optimisation algorithm" Chaos, Solitons & Fractals Vol
% 44 Issue 9, Sept 2011 pp. 710-718 DOI:10.1016/j.chaos.2011.06.004
% -------------------------------------------------------------------------

    % parse some basic parameters for this process
    p = inputParser;
%     rng(125789);

    p = bsAddBasicalParameters(p, length(Lb));
    
    addParameter(p, 'nNest', 25 );      % The number of nests
%     addParameter(p, 'pamax', 1 );        % Discovery rate of alien eggs/solutions
%     addParameter(p, 'pamin', 0.25 );        % Discovery rate of alien eggs/solutions
    addParameter(p, 'beta', 3/2 );      % beta, for levy flight
%     addParameter(p, 'alphamin', 0.02 );         % Step size factor, increase this to decrease step size
%     addParameter(p, 'alphamax', 0.5 );         % Step size factor, increase this to decrease step size
%     addParameter(p, 'pwr', 0.5 );       % Power that step size is reduced by each generation
    addParameter(p, 'nesD', 1 );        % Number of eggs deleted at each generation
    addParameter(p, 'minNests', 10 );
    
    % set default function of generation initial population as @bsGenerateInitialPopulationByLHS
    addParameter(p, 'initialPopulationFcn', @bsGenerateInitialPopulationByRandom );
    addParameter(p, 'interval', 5 );    % for each interval iterations, perform gradient descent algorithm once
    addParameter(p, 'innerMaxIter', 100 ); % the max number iterations for gradient descent algorithm
    addParameter(p, 'optionsForGB', [] );
    
    p.parse(varargin{:});  
    params = p.Results;
    
    nNest = params.nNest;
    
    % call initial function to generate the initial population
    nests = params.initialPopulationFcn(Lb, Ub, nNest);
    
    % Get the current bestNest
    fitness = 10^10 * ones(nNest, 1);
    nDim = length(Lb);
    gradient = zeros(nDim, nNest);
    
%     gradientObjFunc = @(x)(objFunc(x, 0));
    gradientObjFunc = @(x)(objFunc(x, 1));
    [globalMinFVal, globalBestNest, nests, fitness, gradient, K, globalBestNestGradient] = bsGetBestNestWithGradient(gradientObjFunc, nests, nests, fitness, gradient);
    
    nfev = nNest;   %count the number of evaluations of objective functions
    fs = [];
    
    xInit = globalBestNest;
    
    GBOptions = bsCreateGBOptions(length(Lb), ...
        'optimalFunctionTolerance', params.optimalFunctionTolerance, ...
        'maxIter', params.innerMaxIter, ...
        'isSaveMiddleRes', params.isSaveMiddleRes, ...
        'display', 'off', ...
        'optimalF', params.optimalF);
    
    nSameIter = 0;
    trainingPa = zeros(params.maxIter, 2);
    trainingAlpha = zeros(params.maxIter, 2);
    
    %% Starting iterations
    for iter = 1 : params.maxIter
        
        % a) sort the current nests in order of fitness
    
        % First put vectors into matrix form
        piFi = [nests', gradient', fitness];
        % Sort by Fi in assending order
        piFiS = sortrows(piFi, nDim*2+1);
    
        % Decrease number of nests, we only need lots of nests initially to get
        % a good sampling of the objective function
        nNest = max(params.minNests, nNest - params.nesD);
        nests = piFiS(1:nNest, 1:nDim)';
        gradient = piFiS(1:nNest, nDim+1:nDim*2)';
        fitness = piFiS(1:nNest, end);
%         nTop = max(3, round(nNest*ptop));     %%%%%%%%%%%%%%%
    
%         if iter <= 20
%             alpha = 0.05 + randi([1 85])/100;
%         else
%             sortData = sortrows(trainingAlpha(iter-20:iter-1, :), -2);
% %             [~, index] = max(trainingPa(end-20:end, 2));
% %             paLast = trainingPa(index, 1);
%             alphaLast = mean(sortData(1:5, 1));
%             alpha = normrnd(alphaLast, 0.1);
%             
%             if alpha > 0.95
%                 alpha = 0.95;
%             end
%             
%             if alpha < 0.02
%                 alpha = 0.02;
%             end
%         end
        
        % 2. Loop over each Cuckoo which has been discarded
        % Generate new solutions (but keep the current bestNest)
%         newNests = bsGetCuckoos(params.beta, nests, globalBestNest, Lb, Ub, alpha);   
        newNests = bsGetCuckoosByChaos(params.beta, nests, globalBestNest, Lb, Ub); 
        
        % get the bestNest solution among the new solutions
        [bestFVal, bestNest, nests, fitness, gradient, K, nBetter] = bsGetBestNestWithGradient(gradientObjFunc, nests, newNests, fitness, gradient);
        
%         trainingAlpha(iter, :) = [alpha; nBetter];
        
        % Find the bestNest objective so far  
        if bestFVal < globalMinFVal
            globalMinFVal = bestFVal;
            globalBestNest = bestNest;
            nSameIter = 1;
        end
        
        % Update the counter
        nfev = nfev + nNest; 
        
        if iter <= 20
            pa = 0.05 + randi([1 85])/100;
        else
            sortData = sortrows(trainingPa(iter-20:iter-1, :), -2);
%             [~, index] = max(trainingPa(end-20:end, 2));
%             paLast = trainingPa(index, 1);
            paLast = mean(sortData(1:5, 1));
            pa = normrnd(paLast, 0.1);
            
            if pa > 0.95
                pa = 0.95;
            end
            
            if pa < 0.05
                pa = 0.05;
            end
        end
        
        % Discovery and randomization
        newNests = bsEmptyNests(nests, Lb, Ub, pa) ;
        
        
        % get the bestNest solution among the new solutions
        [bestFVal, bestNest, nests, fitness, gradient, K, nBetter] = bsGetBestNestWithGradient(gradientObjFunc, nests, newNests, fitness, gradient);

  
        trainingPa(iter, :) = [pa, nBetter];
        
        % Update the counter
        nfev = nfev + nNest; 
        
        % Update the counter again
        if params.isSaveMiddleRes
            fs = [fs, [nfev; min(bestFVal, globalMinFVal)]];
        end
        
        if mod(iter, params.interval) == 0 && bestFVal < globalMinFVal
%             [bestNest, bestFVal, ~, output] = bsGBSolveByOptions(bestNest, GBOptions);
            [bestNest, bestFVal, ~, output] = bsGBSolveByOptions(gradientObjFunc, bestNest, Lb, Ub, GBOptions);
            nests(:, K) = bestNest;
            fitness(K) = bestFVal;
            nfev = nfev + output.funcCount;
            
            if params.isSaveMiddleRes
                fs = [fs, [nfev; bestFVal]];
            end
        end

        
        % Find the bestNest objective so far  
        if bestFVal < globalMinFVal
            globalMinFVal = bestFVal;
            globalBestNest = bestNest;
            nSameIter = 1;
        else
            nSameIter = nSameIter + 1;
        end
        
        data.fNew = globalMinFVal;
        data.nfev = nfev;
        data.iter = iter;
        
        exitFlag = bsCheckStopCriteria(data, params);
        [data] = bsPlotMidResults(xInit, data, params, Lb, Ub, exitFlag > 0);
        
        if exitFlag > 0
            break;
        end
        
%         data.fOld = globalMinFVal;
%         data.xOld = globalBestNest;
        
    end %% End of iterations

    x = globalBestNest;
    fval = globalMinFVal;
    output.funcCount = nfev;
    output.iterations = iter;
    output.midResults = fs;  
    output.frames = data.frames;
end

% function nest = bsUpdateNest(nests, nNest, nTop, C, a, beta)
%     % Cross with egg outside elite
%     randIndex = randi([nTop+1, nNest]);
%     xi = nests(:, C);
%     xj = nests(:, randIndex);
%     nDim = size(nests, 1);
%     
%     dist = xj - xi;
%     %Multiply distance by a random number
%     dist = dist .* 0.618;%rand(nDim, 1);
%     nest = xi + dist;
% 
%     if bsIsMember(nest, nests, 'columns')
% 
%         step = bsLevy(nDim, 1, beta);
%         nest = xj + a .* step;
%     end
% end

