%% This script is used to show the self-adapting of the 9 control
% parameters of the HAGCS, as a function of generation index 
% Bin She, bin.stepbystep@gmail.com

close all;
clear;


maxIter = 1000000;
dimensions = [100];
nSimulation = 30;
optimalFunctionTolerance = 1e-8;
maxCuckooIteration = 100000;
maxFevs = 500000;

%% Test function sets
benchmarks = {
    
%     @bsSphere, 'Sphere', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     @bsAxisParallelHyperElliptic, 'Axis parallel hyper-ellipsoid', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     @bsZakharov, 'Zakharov', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     @bsRosenbrock, 'Rosenbrock', @(dim)(ones(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     {@bsStochasticRosenbrock, @(dim)(abs(rand(dim, 1)) + 0.5)}, 'Stochastic Rosenbrock', @(dim)(ones(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     {@bsXinSheYang, @(dim)(abs(rand(dim, 1)))}, 'Xin-She Yang', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     @bsAckley, 'Ackley', @(dim)(zeros(dim, 1)), @(dim)(0), [-50 50], dimensions;
%     @bsGriewank, 'Griewank', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
    @bsRastrigin, 'Rastrigin', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions; 
%     {@bsShiftedRastrigin, @(dim)(rand()-0.5)*5}, 'Shifted Rastrigin', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     @bsSchwefel2_22, "Schwefel's P2.22", @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     {@bsShiftedSchwefel1_2WithNoise, @(dim)[randn(dim); (rand()-0.5)*10]}, "Shifted Schwefel's P1.2 with noise", @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions; 
    
};

testMethods = {
    'HAGCS', '2022_50', {dimensions, 5 500, @bsGenerateInitialPopulationByRandom, 100, 200};
};


%% read the configure information for the optimization solver 
nTestMethods = size(testMethods, 1);
nBenchmarks = size(benchmarks, 1);
nProblem = 0;
for iBenchmark = 1 : nBenchmarks
    nProblem = nProblem + length(benchmarks{iBenchmark, 6});
end
nCases = nProblem * nTestMethods;

TestFuncNames = cell(nCases, 1);
MethodNames = cell(nCases, 1);
MethodClassNames = cell(nCases, 1);
Dimensions = zeros(nCases, 1);

SuccessRates = zeros(nCases, 1);

GlobalMinFuncVals = zeros(nCases, 1);

MeanFuncVals = zeros(nCases, 1);
StdFuncVals = zeros(nCases, 1);
MeanNIterVals = zeros(nCases, 1);
StdNIterVals = zeros(nCases, 1);
MeanNFevVals = zeros(nCases, 1);
StdNFevVals = zeros(nCases, 1);
MeanTimeVals = zeros(nCases, 1);
StdTimeVals = zeros(nCases, 1);

iCase = 0;

for iBenchmark = 1 : nBenchmarks
    
    objFuncName = benchmarks{iBenchmark, 2};    
    xRange = benchmarks{iBenchmark, 5};
    dimensions = benchmarks{iBenchmark, 6};
    
    
    for iDim = 1 : length(dimensions)
        
        nDim = dimensions(iDim);
        
        maxFevs = floor(10000 * nDim);
        
        if length(xRange(:)) == 2
            lower = ones(nDim, 1) * xRange(1);
            upper = ones(nDim, 1) * xRange(2);
        else
            lower = xRange(:, 1);
            upper = xRange(:, 2);
        end
        
        if isa(benchmarks{iBenchmark, 3},'function_handle')
            bestX = benchmarks{iBenchmark, 3}(nDim);
        else
            bestX = benchmarks{iBenchmark, 3};
        end
        
        if isa(benchmarks{iBenchmark, 4},'function_handle')
            minFVal = benchmarks{iBenchmark, 4}(nDim);
        else
            minFVal = benchmarks{iBenchmark, 4};
        end
        
        % create initial point
        initValues = zeros(nDim, nSimulation);
        for i = 1 : nSimulation
            initValues(:, i) = xRange(1) + (xRange(2) - xRange(1)) * rand(nDim, 1);
        end

        
        caseNames = cell(1, nTestMethods);
        
%         figure;
        
        for iTestMethod = 1 : nTestMethods
            
            methodClassName = testMethods{iTestMethod, 1};
            methodName = testMethods{iTestMethod, 2};
            parameters = testMethods{iTestMethod, 3};
            caseNames{iTestMethod} = sprintf('%s_%s', methodClassName, methodName);
            
            tic
            
            results = cell(1, nSimulation);
            
            parfor i = 1 : nSimulation
                if isa( benchmarks{iBenchmark, 1}, 'function_handle')
                    objFunc = benchmarks{iBenchmark, 1};
                else
                    fcnpkgs = benchmarks{iBenchmark, 1};
                    fcn1 = fcnpkgs{1};
                    fcn2 = fcnpkgs{2};
                    data = fcn2(nDim);
                    
                    objFunc = @(x,y)(fcn1(x, y, data));
                end
                
                initX = initValues(:, i);   
                
                
                maxNest = parameters{1};
                minNest = parameters{2};
                nHistory = parameters{3};

                initialPopFcn = parameters{4};

                interval = parameters{5};
                innerMaxIter = parameters{6};

                [xOut, funVal, exitFlag, OUTPUT] = bsHAGCSByShe2022(objFunc, lower, upper, ...
                    'maxNests', maxNest, ...
                    'minNests', minNest,...
                    'nHistory', nHistory, ...
                    'display', 'off', ...
                    'initialPopulationFcn', initialPopFcn,...
                    'interval', interval, ...
                    'innerMaxIter', innerMaxIter,...
                    'optimalFunctionTolerance', optimalFunctionTolerance, ...
                    'maxIter', maxCuckooIteration, ...
                    'optimalF', minFVal, ...
                    'isSaveMiddleRes', true, ...
                    'maxFunctionEvaluations', maxFevs);

                nIter = OUTPUT.iterations;
                nfev = OUTPUT.funcCount;
                midResults = OUTPUT.midResults;
                

                
                fprintf('Test function:%s, dimension:%d, using method:%s, simulation:%d nfev:%d...\n', objFuncName, nDim, caseNames{iTestMethod}, i, nfev);

                if (funVal - minFVal) > 1e-4
                    disp(i);
                end

                results{i}.output = OUTPUT;
                results{i}.xout = xOut;
                results{i}.funval = funVal;
                
                
                
                
            end

            iCase = iCase + 1;
            figure;
            set(gcf, 'position', [408         242        1043         718]);
            load colorTbl.mat;
%             colors{2} = colors{4};
%             colors = {colors{1}, colors{4}, colors{2}};
            colors{3} = colors{4};
            parameters = {'p_b', 'p_c', 'CR', '\alpha_2', '\alpha_3', '\alpha_4', 'p_a', 'p_g', '\alpha_1' };
            indecies = [2, 3, 8, 5, 6, 7, 9, 10, 11];
    
            
            for k = 1 : length(indecies)
                index = indecies(k);
                bsSubPlotFit(3, 3, k, 0.96, 0.94, 0.04, 0.06, 0.05, 0.01);
                
                maxX = -inf;
                maxY = -inf;
                num = 0;
                
                matOutput = cell2mat(results);
                tmp = [matOutput.output];
                funcCounts = [tmp.funcCount];
                [~, sortIndex] = bsMaxK(funcCounts, 3);
                
%                 for i = [2, 1, 3]
                for i = 1 : 3
                    output = results{sortIndex(i)}.output;
                    
                    x = 1 : output.iterations;
                    y = output.paramHistory(index, :);
                    
                    if x(end) > maxX
                        maxX = x(end);
                    end
                    
                    if max(y) > maxY
                        maxY = max(y);
                    end
                    
                    plot(x, y, '--', 'color', colors{i}, 'linewidth', 2); hold on;
                    
                    if k <= 6
                        set(gca, 'xtick', [], 'xticklabel', []);
                    end
                end
                
%                 text(maxX*0.9, maxY, parameters{k}, 'fontsize', 13);
                if k > 6
                    xlabel('Generation index $t$', 'fontweight', 'bold', 'interpreter', 'latex');
                else
%                     xlabel();
                end
                
                title(sprintf('(%s) - %s', 'a'+k-1, parameters{k}), 'fontweight', 'bold');
                
                if mod(k, 3) == 1
                    ylabel(sprintf('Parameter value'));
                end
                
                bsSetDefaultPlotSet(bsGetDefaultPlotSet());
            end
        end
        
    end
end





