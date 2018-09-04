% Datfile_convert script
[FileName,PathName,FilterIndex] = uigetfile('*.csv','MultiSelect','on');

for i = 1:length(FileName)
    M = csvread(strcat(PathName,char(FileName(i))));
%%% -Inf to 0
    M(M==-Inf) = 0;

%%% Eliminate the same time data (HW problem)
    temp = [];
    start = 1;
    for j=1:length(M)-1
        if j ~= 1
            if M(j,1) == M(j-1,1)
                if start ~= j
                    temp = cat(1,temp,M(start:j-1,:));
                end
             start = j+1;
            end
        end
        if j == length(M)-1
            temp = cat(1,temp,M(start:j-1,:));
        end
    end

%%% Interpolation for missing data
    time1 = temp(1,1);
    time2 = temp(end,1);
    axis = [time1:0.001:time2];
    ret_val = interp1(temp(:,1),temp(:,2:end),axis,'previous');
    ret_val = cat(2,axis',ret_val);
%%% Save the file
    csvwrite(strcat('input_', char(FileName(i))),ret_val);
end