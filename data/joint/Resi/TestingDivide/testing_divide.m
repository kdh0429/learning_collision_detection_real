fileName = strcat('../','Testing_raw_data_','.csv');
TestingRaw = load(fileName);
T= 100;
for i=1:fix(size(TestingRaw,1)/T)
    newName = strcat('Testing_raw_data_',int2str(i),'.csv');
    newData = TestingRaw(T*(i-1)+1:T*i+1,:);
    csvwrite(newName, newData);
end