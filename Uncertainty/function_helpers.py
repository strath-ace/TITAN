from functools import partial
import numpy as np

# This is a monument in dedication to functools.partial 
class partial_helper():
    def __init__(self):
        pass

    def comparison_multifi(self,callable_datum,callable_of_interest,comparison_method,is_reconstruction,vector):
        output =[]

        # If producing a discrepancy this value is the ground truth, if reconstructing it is the uncorrected model prediction
        datum_values = np.array([callable_datum(vector)]).flatten()

        # If producing a discrepancy this value is the model prediction, if reconstructing it is the correction to be applied
        values_of_interest = np.array([callable_of_interest(vector)]).flatten()

        if not len(datum_values)==len(values_of_interest): raise Exception('Error: Callable output dimensions do not agree!')

        for datum, value_of_interest in zip(datum_values,values_of_interest):
            match comparison_method:
                case 'distance':
                    if not is_reconstruction:
                        output.append(value_of_interest-datum)
                    else:
                        output.append(datum-value_of_interest)
                case 'scale':
                    if not is_reconstruction:
                        output.append(value_of_interest/datum)
                    else:
                        output.append(datum/value_of_interest)
                    
        return np.array(output).flatten()

    def surrogate_HDMR(self,hdmr,output_names,vector):
        output =[]
        for output_name in output_names:
            output.append(hdmr.call_global_surrogate(vector,output_name))
        return np.array(output).flatten()
    
    
    def construct_callable(self,type='HDMR',arguments=[],comparison_method='distance'):
        match type:
            case 'HDMR':
                return partial(self.surrogate_HDMR,*arguments)
            case 'model_discrepancies':
                arguments.append(comparison_method)
                arguments.append(False)
                return partial(self.comparison_multifi,*arguments)
            case 'corrected_surrogate':
                arguments.append(comparison_method)
                arguments.append(True)
                return partial(self.comparison_multifi,*arguments)

            