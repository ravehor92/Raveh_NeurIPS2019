% A function that runs the algorithm in a maze. 
% Maze type 1 - simple, 2 - complex 9X4
% Episodes - Number of episodes (at each one, a random new position is casted, but collected samples remain)
% times - number of parallel runs
% R_max - currently the rewards are 0 and R_max.
% Bell_iterations - fixed number of value iterations.
% Episode_Length - How many steps within an episode.
% maze_size - [Num of rows, Num of columns]. if the maze_type is 2 - then it's necessarily 9X4. 
% rand_a_flag - 0 - use the algorithm. 1 - use a random action. 
% Mom_flag - 0 - use the sample average. 1 - use the median of means. 
% km - number of groups for calculating the median. If km=1, this is jsut a sample average. 
% Noise var - variance of reward noise sigma^2 = a^2/12. (for uniform distributed noise in [-a/2,a/2]). 
function [Total_Discounted_R,std_Total_Discounted_R, Total_visits, num_of_samples,Q]=maze_func_single_agent(Episodes,Num_of_Agents,times,eps_a,eps_b,gamma,R_max,k,Bell_iterations,...
                                               Episode_Length,maze_size,maze_type, rand_a_flag, km, noise_var, noise_type, est_type,m,var_Q,Num_cases, current_case)
 
Total_Discounted_R = zeros(Num_of_Agents, Episodes,times) ;
if (maze_type==2)
	maze_size = [9,4] ; 
end

Total_visits = zeros(maze_size(1),maze_size(2),Num_of_Agents, times) ;
% times 
for tt=1:times
    disp(['Times Number',num2str(tt)]);
    disp(['']);
    disp(['']);disp(['']);disp(['']);pause(1);
%times
Qmax = 100/(1-0.98) ; 

%Building Ka group 
Ka = km ; 

while Ka(end) < k
	Ka = [Ka, Ka(end)*2] ; % CHANGE BACK TO 2
end
if Ka(end) > k 
	Ka(end) = k ; 
end
	
%

A = 4 ; % Number of actions
S =  maze_size(1)*maze_size(2) ; % Number of states for a square maze
U = zeros(S,A,2,k) ; % approximation set for a square maze. Rows - states,Columns - actions, above - (r,s') which are the only 
                       % we need to save, ichiban Above - the samples 
flags = zeros(S,A) ;  % States whether we have a sample already 
num_of_samples = zeros(S,A) ; % Number of samples for each state-action 

%% Policy Execution
states = randi(S,1,Num_of_Agents); % Randomizing initial positions, using vector representation of the maze.
%a_chosen = zeros(1,Num_of_Agents) ; 
Q = Qmax*ones(S,A) ;  % Initiated as the optimistic max.
Q_fN = repmat(Q,1,1,Num_of_Agents) + sqrt(var_Q)*randn(S,A,Num_of_Agents) ; 
Q_noisy = zeros(S,A,Num_of_Agents); 
for ii = 1:Num_of_Agents
	Q_noisy(:,:,ii) = max(0,min(Qmax,Q_fN(:,:,ii))) ;				
end


for Episode = 1:Episodes
	disp(['']);
	disp(['=========================']);
	disp(['Episode Number',num2str(Episode)]);
	disp(['=========================']);
	disp(['']);
	states = initial_state(maze_type, S, Num_of_Agents, maze_size);
	%states = randi(S,1,Num_of_Agents); 
	for Step = 1:Episode_Length
		disp(['Case number ',num2str(current_case),' Out of',num2str(Num_cases)]);
		disp(['Times number ',num2str(tt),' of ',num2str(times),', Episode Number',num2str(Episode)]);
		disp(['Step Number ',num2str(Step)]);
		VI_flag = 0 ; 
		for Agent = 1:Num_of_Agents % Move each of the agents once
			if rand_a_flag == 1 
				a_chosen = randi(4) ;
			else
				[~,a_chosen] = max(Q_noisy(states(Agent),:)) ; 
			end
			s_mat = zeros(1,2) ; 
			[s_mat(1),s_mat(2)] =ind2sub([maze_size(1),maze_size(2)],states(Agent));
			Total_visits(s_mat(1), s_mat(2), Agent, tt) = Total_visits(s_mat(1), s_mat(2), Agent, tt) + 1 ; 
			agent_class = 0 ; 
			if Agent > floor(Num_of_Agents/2)
				agent_class = 1 ; 
			end
			[s_new,R, R_real] = move(noise_var, noise_type, s_mat, a_chosen, maze_type, maze_size,R_max,agent_class) ; 
			s_new = sub2ind([maze_size(1),maze_size(2)],s_new(1),s_new(2)) ;
			if num_of_samples(states(Agent),a_chosen) < k 
				num_of_samples(states(Agent),a_chosen) = num_of_samples(states(Agent),a_chosen) + 1 ;
				U(states(Agent), a_chosen, :, num_of_samples(states(Agent),a_chosen)) = [s_new,R] ; 
				VI_flag = 1; 
			end
			disp(['Agent ',num2str(Agent),' has currently ',num2str(num_of_samples(states(Agent),a_chosen)),' samples. flag = ',num2str(VI_flag)]);
			states(Agent) = s_new ; 
			Total_Discounted_R(Agent,Episode,tt) = Total_Discounted_R(Agent,Episode,tt) + R_real; %R*gamma^(Step-1) ; 
		end
		if VI_flag==1
			Q = VI_fast(Bell_iterations,S,A, num_of_samples, Ka, Qmax, Q, U, gamma, eps_a,eps_b, km,est_type,m) ;
			Q_fN = repmat(Q,1,1,Num_of_Agents) + sqrt(var_Q)*randn(S,A,Num_of_Agents) ; 
			Q_noisy = zeros(S,A,Num_of_Agents); 
			for ii = 1:Num_of_Agents
				Q_noisy(:,:,ii) = max(0,min(Qmax,Q_fN(:,:,ii))) ;				
			end			
		end
		
	end
end
%times
end			
%times				
std_Total_Discounted_R = std(Total_Discounted_R,1,3) ;
Total_Discounted_R = mean(Total_Discounted_R,3) ; 
Total_visits = mean(Total_visits, 4) ;

end








