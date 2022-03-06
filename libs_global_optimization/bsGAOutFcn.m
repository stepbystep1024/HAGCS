function [state,options,optchanged] = bsGAOutFcn(options, state, flag, minFVal, Tol)
     optchanged = false;
     
     i = get(getCurrentTask(),'ID');
     
     switch flag
         case 'init'
             initial_population = state.Population;
             assignin('base', sprintf('initial_population_%d', i), initial_population);
         case 'iter'
             if min(state.Score) - minFVal < Tol
                 optchanged = true;
             end
         case 'done'
         otherwise
     end
 end