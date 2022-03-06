%% This script is used to compare the performance of different global optimization algorithms 
% on different optimization problems (different test problems with different dimensions)
% Author: Bin She
% Affiliation: Huawei
% Email: bin.stepbystep@gmail.com
close all;
clear;

%% Some basica settings such as stop criteria
% the maximum number of iteration, we generally don't use this item for the stop criterion 
maxIter = 1000000;              
% when f - f* < optimalFunctionTolerance, iteration will be stopped. f* is
% the global minimum function value
optimalFunctionTolerance = 1e-8;
% the number of simulations (for each problem) you'd like to run
nSimulation = 10;
% the dimension of the problems you'd like to test. 
% dimensions = 10:10:100;
dimensions = 30;
% whether to save the middle result of in each simulation
isSaveMiddleRes = false;
% no need to trace the convergence paths, it will be set to true only when
% we need to watch the animation, see testShowAnimation.m
isSaveDetailUpdates = false;
%% Test function sets
benchmarks = {    
    %1. the objective function handle. It looks like [f, g] = obj(x, isGradient)
    %   see bsSphere.m for example. 
    %   Note that in some cases, for instance, the CF2 function, it needs
    %   to randomly generate a objective function, so the first item looks like
    %   a cell data {objective function, extra data to construct the ojbective function}
    %2. the name of this objective function
    %3. the global optimal solution. It is actually a funciton handle which
    %   returns a vector (\mathbf{x}_{optimal}) by given the dimension of the problem
    %4. the global minimum objecteve function value (y) of the problem
    %5. the range of the parameters
    %6. the dimensions (that will be tested) of the test function
    
%     @bsSphere, 'f1: Sphere', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
    @bsSchwefel2_22, "f2: Schwefel's P2.22", @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
    @bsRosenbrock, 'f3: Rosenbrock', @(dim)(ones(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     {@bsCFSingle, @(dim)(abs(rand(3, 1)+0.1))}, 'f4: CF1', @(dim)(ones(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     {@bsStochasticRosenbrock, @(dim)(abs(rand(dim, 1)))}, 'f5: Stochastic Rosenbrock', @(dim)(ones(dim, 1)), @(dim)(0), [-100 100], dimensions;
    @bsAckley, 'f6: Ackley', @(dim)(zeros(dim, 1)), @(dim)(0), [-50 50], dimensions;
    @bsGriewank, 'f7: Griewank', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     @bsRastrigin, 'f8: Rastrigin', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions; 
    {@bsShiftedRastrigin, @(dim)(rand(dim, 1)-0.5)*50}, 'f9: Shifted Rastrigin', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     @bsSchwefel, 'f10: Schwefel', @(dim)(420.9687*ones(dim, 1)), @(dim)(0), [-500 500], dimensions;
    {@bsStochasticGriewank, @(dim)(abs(rand(dim, 1)))}, 'f11: Stochastic Griewank', @(dim)(zeros(dim, 1)), @(dim)(0), [-100 100], dimensions;
%     {@bsCFComplex, @(dim)([abs(rand(5, 1)+0.1)])}, 'f12: CF2', @(dim)(ones(dim, 1)), @(dim)(0), [-100 100], dimensions;
};

% a function handle to generate the size of initial population
% NInitFcn = @(nDim)(nDim+50);

% test methods
testMethods = {
    %1. the name of the method
    %2. note of the method
    %3. special parameters of the corresponding method
%     'PSO', 'MATLAB', {100};     % 100 refers to the initial population size of the PSO 
%     'GA', 'MATLAB', {100};      % 100 refers to the initial population size of the PSO 
    
    % the special parameters of the following 2 methods are 
    % NInit
    % p_a (switching parameters)
    % alpha
    % population initial function
    'CS', 'Yang and Deb', {15 0.75 0.05, @bsGenerateInitialPopulationByRandom}; 
    'GBCS', 'Fateen', {15, 0.75, 0.05, @bsGenerateInitialPopulationByRandom};
    
    % the special parameters are
    % NInit, 
    % p_a (switching parameters)
    % alpha
    % population initial function
    % interval, for each interval iterations, perform gradient descent algorithm once
    % the max iteration number of inner loop
    'AGBCS', 'She et al.', {15, 0.75, 0.05, @bsGenerateInitialPopulationByRandom, 50, 200};
    
    % the special parameters are 
    % maximum number of nests exists in the population
    % the mimimum number of nests exists in the population
    % the number of history iterations saved for generating new control parameters
    'AHSACS', 'She et al.', {150, 5, 500, @bsGenerateInitialPopulationByRandom};
    
    % call function bsAHSACSByShe2022.m proposed by our paper
    % the special parameters are 
    % maximum number of nests exists in the population
    % the mimimum number of nests exists in the population
    % the number of history iterations saved for generating new control parameters
    % the number of calls of gradient-based local optimization (GBLO) during the optimization process
    % the maximum number iterations of the inner GBLO subroutine 
    'HAGCS', ' (NI=50)', {50, 5, 500, @bsGenerateInitialPopulationByRandom, 50, 200};
    
    'MSACS', 'Gao et al.', {30, 0.75 0.05, @bsGenerateInitialPopulationByRandom};
    'CSAPC', 'Wei et al.', {15, 0.75 0.05, @bsGenerateInitialPopulationByRandom};
};

%% set the optSolver information
optFunc = @bsOptLBFGS;
funcName = func2str(optFunc);

%% read the configure information for the optimization solver 
nTestMethods = size(testMethods, 1);    % the number of methods
nBenchmarks = size(benchmarks, 1);      % the number of test functions
nProblem = 0;                           % the number of problems (all cases of dimensions of all test functions)
for iBenchmark = 1 : nBenchmarks
    nProblem = nProblem + length(benchmarks{iBenchmark, 6});
end
nCases = nProblem * nTestMethods;       % the number of cases

% the following cells are used for constructing a table showing the
% comparions of different algorithms on different problems
TestFuncNames = cell(nCases, 1);        % names of test function in each case
MethodNotes = cell(nCases, 1);          % the author information in each case
MethodNames = cell(nCases, 1);          % the method name in each case
Dimensions = zeros(nCases, 1);          % the dimension of each case
SuccessRates = zeros(nCases, 1);        % the success rate of each case
GlobalMinFuncVals = zeros(nCases, 1);   % the true global minimum f*
MeanFuncVals = zeros(nCases, 1);        % the average f
StdFuncVals = zeros(nCases, 1);         % the standard variation of f
BestFuncVals = zeros(nCases, 1);        % the best f
MeanNIterVals = zeros(nCases, 1);       % the average number of iterations NIter
StdNIterVals = zeros(nCases, 1);        % the standard variation of NIter
BestNIterVals = zeros(nCases, 1);       % the best NIter
MeanNFevVals = zeros(nCases, 1);        % the averate number of function evaluations NFev
StdNFevVals = zeros(nCases, 1);         % the standard variation of NFev
BestFevVals = zeros(nCases, 1);         % the best NFev
MeanTimeVals = zeros(nCases, 1);        % the averate computational time (s)
StdTimeVals = zeros(nCases, 1);         % the standard variation of computational time
BestTimeVals = zeros(nCases, 1);        % the best computational time


groupResults = cell(nCases, 1);
detailResults = cell(nBenchmarks, nTestMethods, nSimulation);

iCase = 0;                              % iteration index indicating the index of current problem case
for iBenchmark = 1 : nBenchmarks
    % for each test function
    
    objFuncName = benchmarks{iBenchmark, 2};    % name of the test function
    xRange = benchmarks{iBenchmark, 5};         % range of the test function
    dimensions = benchmarks{iBenchmark, 6};     % dimensions of the test function
    
    
    for iDim = 1 : length(dimensions)
        % for each dimension case of the current test function
        
        nDim = dimensions(iDim);                % the dimension of the current problem
        
        % set the maximum number of funciton evaluations as 10^4*nDim 
        maxFevs = floor(10000 * nDim);      
        
        % the the lower and upper boundary
        if length(xRange(:)) == 2
            % in this case xRange loos like [lower, upper], lower and upper are the
            % same for all dimension of the parameters
            lower = ones(nDim, 1) * xRange(1);
            upper = ones(nDim, 1) * xRange(2);
        else
            % in this case xRange loos like [lower, upper], lower and upper
            % are two vectors
            lower = xRange(:, 1);
            upper = xRange(:, 2);
        end
        
        % obtain the global optimum solution of the current problem
        if isa(benchmarks{iBenchmark, 3},'function_handle')
            bestX = benchmarks{iBenchmark, 3}(nDim);
        else
            bestX = benchmarks{iBenchmark, 3};
        end
        
        % obtain the global minimum of the current problem
        if isa(benchmarks{iBenchmark, 4},'function_handle')
            minFVal = benchmarks{iBenchmark, 4}(nDim);
        else
            minFVal = benchmarks{iBenchmark, 4};
        end
       
        caseNames = cell(1, nTestMethods);
        
        for iTestMethod = 1 : nTestMethods
            % for each method to solve the curretn problem 
            
            methodName = testMethods{iTestMethod, 1};   % method name
            methodNote = testMethods{iTestMethod, 2};   % method note
            parameters = testMethods{iTestMethod, 3};   % special parameters of the current method
            methodInfo = sprintf('%s_%s', methodName, methodNote);
            
            tic
            
            % the following data are used to record the optimization result
            % of each simulation, so that we can calcuate the statistical
            % performance of each method
            groupFks = zeros(1, nSimulation);
            groupNIter = zeros(1, nSimulation);
            groupNFev = zeros(1, nSimulation);
            groupTime = zeros(1, nSimulation);
            
            parfor i = 1 : nSimulation
                % for each simulation of the problem

                % prepare the objective function
                if isa( benchmarks{iBenchmark, 1}, 'function_handle')
                    objFunc = benchmarks{iBenchmark, 1};
                else
                    % in this case, the objective funciton needs extra
                    % data, for exameple, see the CF2 test function
                    fcnpkgs = benchmarks{iBenchmark, 1};
                    fcn1 = fcnpkgs{1};
                    fcn2 = fcnpkgs{2};
                    data = fcn2(nDim);
                    
                    objFunc = @(x, y)(fcn1(x, y, data));
                end
                
                tic
                
                [xOut, funVal, exitFlag, OUTPUT] = bsRunOneMethodOneSimulation( ...
                    objFunc, ...
                    methodName, ... 
                    parameters, ...
                    lower, ...
                    upper, ...
                    optimalFunctionTolerance, ...
                    maxIter, ...
                    minFVal, ...
                    maxFevs, ...
                    isSaveMiddleRes, ...
                    isSaveDetailUpdates);       

                
                nIter = OUTPUT.iterations;      % get the number of iterations
                nfev = OUTPUT.funcCount;        % get the number of function evaluations
                
                % print some information of the current simulation
                fprintf('Test function:%s, dimension:%d, using method:%s, simulation:%d nfev:%d...\n', objFuncName, nDim, methodInfo, i, nfev);
                
                
                if (funVal - minFVal) > optimalFunctionTolerance
                    % if this simulation is failed to find the global
                    % minimum
                    fprintf('\tThe %d-th simulation is failed to find the global minimum!!!\n', i);
                end

                % save the results of this simulation
                groupFks(i) = funVal;
                groupNIter(i) = nIter;
                groupNFev(i) = nfev;
                groupTime(i) = toc;
                
                if isSaveMiddleRes
                    detailResults{iBenchmark, iTestMethod, i}.output = OUTPUT;
                    detailResults{iBenchmark, iTestMethod, i}.xout = xOut;
                    detailResults{iBenchmark, iTestMethod, i}.funval = funVal;
                end
            end

            iCase = iCase + 1;
            % save the information into the cells
            TestFuncNames{iCase} = objFuncName;
            MethodNotes{iCase} = methodNote;
            MethodNames{iCase} = methodName;
            Dimensions(iCase) = nDim;
            GlobalMinFuncVals(iCase) = minFVal;
            
            nSuccess = length(find(abs(groupFks - minFVal) <= optimalFunctionTolerance)); 
            SuccessRates(iCase) = nSuccess / nSimulation * 100;
            
            % statistical information of objective function values 
            MeanFuncVals(iCase) = mean(groupFks);
            StdFuncVals(iCase) = std(groupFks);
            BestFuncVals(iCase) = min(groupFks);
            
            
            
            % statistical information of the number of iterations
            MeanNIterVals(iCase) = mean(groupNIter);
            StdNIterVals(iCase) = std(groupNIter);
            BestNIterVals(iCase) = min(groupNIter);
            
            % for all the simulations that are failed to find the global
            % miminum, we set their function evaluations as the maximum
            % function evaluations 10^4*nDim
            FailedIndex = find(abs(groupFks - minFVal) > optimalFunctionTolerance);
            groupNFev(FailedIndex) = maxFevs;
            
            % statistical information of the numbmer of function evaluations
            MeanNFevVals(iCase) = mean(groupNFev);
            StdNFevVals(iCase) = std(groupNFev);
            BestFevVals(iCase) = min(groupNIter);
            
            % statistical information of computational time (in s)
            MeanTimeVals(iCase) = mean(groupTime);
            StdTimeVals(iCase) = std(groupTime);
            BestTimeVals(iCase) = min(groupTime);

            % construct the talbe and print it in console
            T = table(TestFuncNames, MethodNotes, MethodNames, Dimensions, SuccessRates, ...
                    MeanFuncVals, StdFuncVals, BestFuncVals, ...
                    MeanNFevVals, StdNFevVals, BestFevVals, ...
                    MeanNIterVals, StdNIterVals, BestNIterVals, ...
                    MeanTimeVals, StdTimeVals, BestTimeVals)

            % also save the results into results
            groupResults{iCase}.groupFks = groupFks;
            groupResults{iCase}.groupNIter = groupNIter;
            groupResults{iCase}.groupNFev = groupNFev;
            groupResults{iCase}.groupTime = groupTime;
            
            
        end
        
        try
            save Table_Compare_Methods_Results_2022_03_06;
        catch
            save Table_Compare_Methods_Results_2022_03_06 -v7.3;
        end
    end
    
    
end





