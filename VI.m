% VI.m - Value Iteration. Performs value iterations until the error is smaller than eps_a, 
%        or until a maximum number of Num_iterations steps has been performed
% Num_iterations - maximal number of iterations to perform
% S - a vector representing the maze
% A - a vector representing the number of actions
% num_of_samples - number of samples for each state-actions 
% Ka - a vector containing the possible values of active samples numbers 
% Qmax - the maximal possible value for the Q function 
% Q - the given Q function, a [S,A] matrix
% U - the given approximation set (samples are inside)
%     approximation set for a square maze. Rows - states,Columns - actions, 
%     3rd dimension - (r,s') 4th dimension - the samples 
% gamma - discount factor
% eps_a - maximal allowed Bellman error
% eps_b - exploration factor
% km - number of active samples

% Q_new - the resulting Q function, a [S,A] matrix 
function [Q_new]=VI(Num_iterations,S,A, num_of_samples, Ka, Qmax, Q, U, gamma, eps_a, eps_b, km)

Q_new=Q ; Bellman_Iterations = 1 ; 
Q_BQ_diff = eps_a+1 ; 
while ((Bellman_Iterations <= Num_iterations)&&(Q_BQ_diff>eps_a)) % Perform a Bellman iteration
	Q = Q_new ; 
	for s_bellman = 1:S
		for a_bellman = 1:A 
			k_active = Active_Samples(num_of_samples(s_bellman, a_bellman),Ka, km) ; % number of active samples
			if k_active == 0 
				Q_new(s_bellman,a_bellman) = Qmax ; 				
			elseif k_active > 0  				
				[~,a_chosen] = max(Q(U(s_bellman,a_bellman,1,1:k_active),:),[],2) ;
				
				nig = ceil(k_active/km) ; % num of samples in a subgroup
				avg_vec	= zeros(1,km) ; % Each value is the average for a sub-group			
				for i = 1: km-1 
					s_indices = squeeze(U(s_bellman,a_bellman,1,((i-1)*nig+1):i*nig)) ; 
					a_indices = a_chosen(((i-1)*nig+1):i*nig) ;
					avg_vec(i) = mean(squeeze(U(s_bellman,a_bellman,2,((i-1)*nig+1):i*nig)) + gamma*Q(sub2ind(size(Q),s_indices, a_indices))) ;
				end
				% Calcualting for the last group
				s_indices = squeeze(U(s_bellman,a_bellman,1,((km-1)*nig+1):k_active)) ; 
				a_indices = a_chosen(((km-1)*nig+1):k_active) ;
				avg_vec(km) = mean(squeeze(U(s_bellman,a_bellman,2,((km-1)*nig+1):k_active)) + gamma*Q(sub2ind(size(Q),s_indices, a_indices))) ;
							
				F = median(avg_vec) ;
				F = (eps_b/sqrt(k_active)) + F ;
				Q_new(s_bellman,a_bellman) = max(0,min(Qmax,F)) ; 				
			end
		end
	end 
	Bellman_Iterations = Bellman_Iterations+1; 
	Q_BQ_diff = abs(max(max(Q-Q_new))) ; 
end 
end
