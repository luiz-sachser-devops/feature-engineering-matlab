%% Script for Assignment 1 ['penguins_size.csv']
%% Script Task 1
data_final = readtable('penguins_size.csv');
%{
NOTES:
 -> Reads the CSV file
 -> Store into a table
%}

%% Script Task 2
data_missing = ismissing(data_final);
fprintf('Total missing values: %d \n',sum(sum(data_missing)))
columns = data_final.Properties.VariableNames;
for col = columns
    miss = ismissing(data_final(:,col));
    sm_miss = sum(miss);
    format = 'Column "%s" has %d missing values\n';
    fprintf(format,string(col),sm_miss)
end
%{
NOTES:
 -> Find how many missing values are on each column
 -> Display total missing values by column and the total
%}

%% Script Task 3
means = mean(data_final{:,3:6},'omitnan');
disp(means);
%{
NOTES:
 -> Find the mean value for each column, ommiting Nan values, except the last one
 -> Display mean values for each column
%}

sex = categorical(table2array(data_final(:,'sex')));
f = figure('Name','Distribution of sex','NumberTitle','off');
a = axes('Parent',f);
hold(a,'on');
f3 = histogram(sex,'FaceColor','b','BarWidth',0.5,'LineWidth',0.7,...
    'EdgeColor','w');
title('Sex','FontName','Times New Roman');
box(a,'on');
set(a,'FontName','Times New Roman','FontSize',12);
%{
NOTES:
 -> Plot Female/Male occurences histogram
 -> Display total values for each
%}

for i=1:height(data_final)
    for k=3:6
        if ismissing(data_final(i,k))
            data_final(i,k) = num2cell(means(k-2));
        end
    end
end
%{
NOTES:
 -> Find missing values and replace by 'means'
%}

% Fill missing "sex" values
sex = table2array(data_final(:,'sex'));
empty_sex = data_missing(:,end);
sex(empty_sex) = {'MALE'};
% data = [data(:,1:end-1) table(sex,'VariableNames',{'sex'})];
data_final(:,7) = table(sex,'VariableNames',{'sex'});
%{
NOTES:
 -> Find missing values and replace by 'MALE'
%}


%% Script Task 4
[species, Vec_species] = grp2idx(data_final.species);
[island, Vec_island] = grp2idx(data_final.island);
[sex, Vec_sex] = grp2idx(data_final.sex);
%{
NOTES:
 -> Create vectors of group indexes for "species", "island" and "sex"
%}

clean_data = data_final(:,:);
clean_data.species = species;
clean_data.island = island;
clean_data.sex = sex;

mat_data = table2array(clean_data);
save("data.mat","mat_data")
save("species_mapping.mat","Vec_species")
save("island_mapping.mat","Vec_island")
save("sex_mapping.mat","Vec_sex")
%{
NOTES:
 -> Clean the "data_final table"
 -> Save duplicated workspace and categorical vectors
%}

X=mat_data;

%%  Script Task 5 - Clustering data & Plot Silhouette

% Baseline: all columns
f1 = figure('Name','Plot Silhouette','NumberTitle','off');

plots = subplot(1,3,1);
hold(plots,'on');

Xk = X(:,[1,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Species and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',1)
grid on;
xlim([0, 1]);


plots = subplot(1,3,2);
hold(plots,'on');

Xk = X(:,[2,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Island and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',1)
grid on;
xlim([0, 1]);


plots = subplot(1,3,3);
hold(plots,'on');

Xk = X(:,[7,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Sex and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',1)
grid on;
xlim([0, 1]);


%% Script Task 6 - Outliers detection - Whisker plot

f4 = figure('Name','Species Whisker Plot','NumberTitle','off');

a = subplot(2,2,1);
hold(a,'on');
boxplot(X(:,3), X(:,1),'OutlierSize',10,'Symbol','*');
title 'Species X Culmen Length';
xlabel 'Species'; 
ylabel 'Culmen Length (mm)';
 
a = subplot(2,2,2);
hold(a,'on');
boxplot(X(:,4), X(:,1),'OutlierSize',10,'Symbol','*');
title 'Species X Culmen Depth';
xlabel 'Species'; 
ylabel 'Culmen Depth (mm)';

a = subplot(2,2,3);
hold(a,'on');
boxplot(X(:,5), X(:,1),'OutlierSize',10,'Symbol','*');
title 'Species X Flipper Length';
xlabel 'Species'; 
ylabel 'Flipper Length (mm)';

a = subplot(2,2,4);
hold(a,'off');
boxplot(X(:,6), X(:,1),'OutlierSize',10,'Symbol','*');
title 'Species X Body Mass';
xlabel 'Species'; 
ylabel 'Body Mass (g)';

f5 = figure('Name','Island Whisker Plot','NumberTitle','off');

a = subplot(2,2,1);
hold(a,'on');
boxplot(X(:,3), X(:,2),'OutlierSize',10,'Symbol','*');
title 'Island X Culmen Length';
xlabel 'Island'; 
ylabel 'Culmen Length (mm)';
 
a = subplot(2,2,2);
hold(a,'on');
boxplot(X(:,4), X(:,2),'OutlierSize',10,'Symbol','*');
title 'Island X Culmen Depth';
xlabel 'Island'; 
ylabel 'Culmen Depth (mm)';

a = subplot(2,2,3);
hold(a,'on');
boxplot(X(:,5), X(:,2),'OutlierSize',10,'Symbol','*');
title 'Island X Flipper Length';
xlabel 'Island'; 
ylabel 'Flipper Length (mm)';

a = subplot(2,2,4);
hold(a,'off');
boxplot(X(:,6), X(:,2),'OutlierSize',10,'Symbol','*');
title 'Island X Body Mass';
xlabel 'Island'; 
ylabel 'Body Mass (g)';


f6 = figure('Name','Sex Whisker Plot','NumberTitle','off');

a = subplot(2,2,1);
hold(a,'on');
boxplot(X(:,3), X(:,7),'OutlierSize',10,'Symbol','*');
title 'Sex X Culmen Length';
xlabel 'Sex'; 
ylabel 'Culmen Length (mm)';
 
a = subplot(2,2,2);
hold(a,'on');
boxplot(X(:,4), X(:,7),'OutlierSize',10,'Symbol','*');
title 'Sex X Culmen Depth';
xlabel 'Sex'; 
ylabel 'Culmen Depth (mm)';

a = subplot(2,2,3);
hold(a,'on');
boxplot(X(:,5), X(:,7),'OutlierSize',10,'Symbol','*'); 
title 'Sex X Flipper Length';
xlabel 'Sex'; 
ylabel 'Flipper Length (mm)';

a = subplot(2,2,4);
hold(a,'off');
boxplot(X(:,6), X(:,7),'OutlierSize',10,'Symbol','*');
title 'Sex X Body Mass';
xlabel 'Sex'; 
ylabel 'Body Mass (g)';

%% Script Task 7 - Standard Deviation

% Set the standard deviation factor
Std_factor = 2;

% Calculate the upper and lower limits for each column
for i = 1:size(X, 2)
    column = X(:, i);
    Upper_limit = mean(column) + std(column) * Std_factor;
    Lower_limit = mean(column) - std(column) * Std_factor;
    
    % Determine which data points are outliers
    No_outlier = ((column > Upper_limit) | (column < Lower_limit));
    
    % Print the number of outliers in this column
    fprintf('Number of outliers in column %d: %d\n', i, sum(No_outlier));
end

%% Script Task 8 - Log Plot Silhouette

% Baseline: all columns
f11 = figure('Name','Log Plot Silhouette','NumberTitle','off');

plots = subplot(1,3,1);
hold(plots,'on');

XLog = X;
XLog(:,[3:6]) = log(XLog(:,[3:6]));

Xk = XLog(:,[1,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Species and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',0.9)
grid on;
xlim([0, 1]);

plots = subplot(1,3,2);
hold(plots,'on');

Xk = XLog(:,[2,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Island and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',0.9)
grid on;
xlim([0, 1]);

plots = subplot(1,3,3);
hold(plots,'on');

Xk = XLog(:,[7,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Sex and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',0.9)
grid on;
xlim([0, 1]);

%% Script Task 8 - Plot Silhouette Euclidean Norm

figure1 = figure('Name','Plot Silhouette with Euclidean Norm','NumberTitle','off');


plots = subplot(1,3,1);
hold(plots,'on');

XNorm = normalize(X); % normalize the data
Xk = XNorm(:,[1,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Species and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',0.9)
grid on;
xlim([0, 1]);

plots = subplot(1,3,2);
hold(plots,'on');

XNorm = normalize(X); % normalize the data
Xk = XNorm(:,[2,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Island and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',0.9)
grid on;
xlim([0, 1]);

plots = subplot(1,3,3);
hold(plots,'on');

XNorm = normalize(X); % normalize the data
Xk = XNorm(:,[7,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Sex and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',0.9)
grid on;
xlim([0, 1]);

%% Script Task 9 - Plot Silhouette Min Max
figure2 = figure('Name','Plot Silhouette with Min-Max','NumberTitle','off');

plots = subplot(1,3,1);
hold(plots,'on');

XNorm = (X - min(X)) ./ (max(X) - min(X)); % min-max scaling
Xk = XNorm(:,[1,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Species and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',0.9)
grid on;
xlim([0, 1]);

plots = subplot(1,3,2);
hold(plots,'on');

XNorm = (X - min(X)) ./ (max(X) - min(X)); % min-max scaling
Xk = XNorm(:,[2,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Island and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',0.9)
grid on;
xlim([0, 1]);

plots = subplot(1,3,3);
hold(plots,'on');

XNorm = (X - min(X)) ./ (max(X) - min(X)); % min-max scaling
Xk = XNorm(:,[7,3:6]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Sex and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',0.9);
grid on;
xlim([0, 1]);


%% Script Task 10 - Plot New Feature

Xcopy = X;

new_column = zeros(344,1);
Xcopy = [Xcopy, new_column];
Xcopy(:, 8) = Xcopy(:, 5) ./ Xcopy(:, 6) * 100;
Xcopy(:, 8) = log(Xcopy(:, 8));

newFigure1 = figure('Name','Body Flipper Ratio Plot Silhouette','NumberTitle','off');

plots = subplot(1,3,1);
hold(plots,'on');

Xk = Xcopy(:,[1,3,4,8]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Species and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',1)
grid on;
xlim([0, 1]);


plots = subplot(1,3,2);
hold(plots,'on');

Xk = Xcopy(:,[2,3,4,8]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Island and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',1)
grid on;
xlim([0, 1]);


plots = subplot(1,3,3);
hold(plots,'on');

Xk = Xcopy(:,[7,3,4,8]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Sex and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',1)
grid on;
xlim([0, 1]);


% New Feature culmen area

Xcopy = [Xcopy, new_column];
Xcopy(:, 9) = Xcopy(:, 3) .* Xcopy(:, 4);
Xcopy(:, 9) = log(Xcopy(:, 9));


newFigure2 = figure('Name','Culmen Area Plot Silhouette','NumberTitle','off');

plots = subplot(1,3,1);
hold(plots,'on');

Xk = Xcopy(:,[1,3,4,9]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Species and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',1)
grid on;
xlim([0, 1]);


plots = subplot(1,3,2);
hold(plots,'on');

Xk = Xcopy(:,[2,3,4,9]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Island and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',1)
grid on;
xlim([0, 1]);


plots = subplot(1,3,3);
hold(plots,'on');

Xk = Xcopy(:,[7,3,4,9]); 
k = 2;
idx = kmeans(Xk,k);
[s,h] = silhouette(Xk,idx,'sqEuclidean');
title 'Sex and attributes';
hold on;
y = ylim; % current y-axis limits
x = mean(s);
xlabel(['s= ' num2str(x)]);
plot([x x],[y(1) y(2)],'r--','LineWidth',1)
grid on;
xlim([0, 1]);

%% Script Task 11 - Graph Best Features

% Species data
graph1 = figure('Name','Best Features - Species','NumberTitle','off');

data = Xcopy;

fsX = data(:,[3,4,5,6,8,9]);
fsY = data(:,[1]);

[idx,scores] = fscchi2(fsX,fsY);
[sorted_scores, sorted_idx] = sort(scores, 'descend');

% Create a bar chart of the top 4 feature scores
bar(sorted_scores(1:4))

% Get the names of the top 4 features
feature_names = {'Culmen Length', 'Culmen Depth', 'Flipper Length', 'Body Mass', 'Flipper x Body Ratio', 'Culmen Area'};
top_feature_names = feature_names(sorted_idx(1:4));

% Set the x-axis tick labels to the top 4 feature names
set(gca, 'xticklabel', top_feature_names)

% Add axis labels and a title
xlabel('Features')
ylabel('Feature Score')
title('Best Features - Species (Top 4)')

% Island data
graph2 = figure('Name','Best Features - Island','NumberTitle','off');



data = Xcopy;

fsX = data(:,[3,4,5,6,8,9]);
fsY = data(:,[2]);

[idx,scores] = fscchi2(fsX,fsY);
[sorted_scores, sorted_idx] = sort(scores, 'descend');

% Create a bar chart of the top 4 feature scores
bar(sorted_scores(1:4))

% Get the names of the top 4 features
feature_names = {'Culmen Length', 'Culmen Depth', 'Flipper Length', 'Body Mass', 'Flipper x Body Ratio', 'Culmen Area'};
top_feature_names = feature_names(sorted_idx(1:4));

% Set the x-axis tick labels to the top 4 feature names
set(gca, 'xticklabel', top_feature_names)

% Add axis labels and a title
xlabel('Features')
ylabel('Feature Score')
title('Best Features - Island (Top 4)')



graph3 = figure('Name','Best Features - Sex','NumberTitle','off');

data = Xcopy;

fsX = data(:,[3,4,5,6,8,9]);
fsY = data(:,[7]);

[idx,scores] = fscchi2(fsX,fsY);
[sorted_scores, sorted_idx] = sort(scores, 'descend');

% Create a bar chart of the top 4 feature scores
bar(sorted_scores(1:4))

% Get the names of the top 4 features
feature_names = {'Culmen Length', 'Culmen Depth', 'Flipper Length', 'Body Mass', 'Flipper x Body Ratio', 'Culmen Area'};
top_feature_names = feature_names(sorted_idx(1:4));

% Set the x-axis tick labels to the top 4 feature names
set(gca, 'xticklabel', top_feature_names)

% Add axis labels and a title
xlabel('Features')
ylabel('Feature Score')
title('Best Features - Sex (Top 4)')