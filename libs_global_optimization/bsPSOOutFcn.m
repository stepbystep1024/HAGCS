function stop = bsPSOOutFcn(optimValues, state, minFVal, Tol)
     stop = false;
 
   
     switch state
         case 'init'
         case 'iter'
             if optimValues.bestfval - minFVal < Tol
                 stop = true;
             end
         case 'done'
             
         otherwise
     end
     
 end