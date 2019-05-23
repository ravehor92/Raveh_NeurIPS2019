% move.m  - given a state and action, the function returns the next state and the reward. 
%           Simple wrapping around maze with a reward R_max on the top right corner, with all other states having reward 0. 
% s - the current state in the maze, of the form [m,n] . 
% a - the current action in the maze, a number between 1-4. 
%     1 - up, 2 - down, 3 - right, 4 - left.
% maze_size - the size of a simple maze. [M,N] means M rows and N columns. 

function [s_new,R] = move(s, a, maze_size,R_max)

s_new = zeros(1,2) ; 
if s == [1,maze_size(2)]
	s_new(1) = randi(maze_size(1)) ; 
	s_new(2) = randi(maze_size(2)) ; 
	R = R_max ; 
else
	switch a 
		case 1 
			s_new(1) = s(1) - 1 +maze_size(1)*(s(1)==1); 
			s_new(2) = s(2) ; 
		case 2
			s_new(1) = s(1) + 1 - maze_size(1)*(s(1)==maze_size(1)); 
			s_new(2) = s(2) ;
		case 3
			s_new(1) = s(1) ; 
			s_new(2) = s(2) + 1 - maze_size(2)*(s(2)==maze_size(2)); 
		otherwise 
			s_new(1) = s(1) ; 
			s_new(2) = s(2) - 1 + maze_size(2)*(s(2)==1); 
		end
	R = 0 ; 
end

end