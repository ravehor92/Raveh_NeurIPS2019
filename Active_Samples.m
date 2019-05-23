% Active_Samples.m - A function that takes an index k1 and a group Ka, 
%                    and finds the lowest index in Ka that is greater or equal to k1. 

function k_active = Active_Samples(k1, Ka, km)
	[~,index] = min(abs(Ka-k1)) ; 
	if k1<km
		k_active = 0 ; 
	elseif Ka(index) > k1
		index = index - 1 ;
		k_active = Ka(index) ;		
	else
		k_active = Ka(index) ;
	end
end