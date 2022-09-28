classdef ExtraTreeNode < handle
    %DECISION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        father
        left
        right
        axis
        value
        bestFeature
        bestPartValue
        mark
    end
    methods
        function obj = ExtraTreeNode()
        end

        function out = copy(obj)
            out = ExtraTreeNode();
            if ~isempty(obj.father)
            out.father = obj.father.copy();
            end
            if ~isempty(obj.left)
            out.left = obj.left.copy();
            end
            if ~isempty(obj.right)
            out.right = obj.right.copy();
            end
            out.axis = obj.axis;
            out.value = obj.value;
            out.bestFeature = obj.bestFeature;
            out.bestPartValue = obj.bestPartValue;
            out.mark = obj.mark;
        end
        
        function  fit(obj,x,y)
            dataset = [x,y];
            SIZE = size(x);
            baseEntropy = calcShannonEnt(y);
            bestInfoGain = 0;
            for i =1:SIZE(2)
                data = dataset(:,i);
                uniqueVals = unique(data);
                sortedUniqueVals = sort(uniqueVals);
                if length(sortedUniqueVals) == 1
                    continue
                end
                j = randi(length(sortedUniqueVals)-1,1);
                bestPartValuei = (sortedUniqueVals(j) + sortedUniqueVals(j+1))/2;
                [dataSetLeft,dataSetRight ]= splitDataset(dataset, i, bestPartValuei);
                probLeft = size(dataSetLeft,1) / SIZE(1);
                probRight = size(dataSetRight,1) / SIZE(1);
                minEntropy = probLeft * calcShannonEnt(dataSetLeft) + probRight * calcShannonEnt(dataSetRight);

                infoGain = baseEntropy - minEntropy;
                if infoGain >= bestInfoGain
                    bestInfoGain = infoGain;
                    obj.bestFeature = i;
                    obj.bestPartValue = bestPartValuei;
                end
            end
            Left_idx = dataset(:,obj.bestFeature) < obj.bestPartValue;
            Right_idx = dataset(:,obj.bestFeature) >= obj.bestPartValue;
            if mean(y( Left_idx)==-1) < mean(y( Right_idx)==-1)
                obj.mark = 1;
            else
                obj.mark = -1;
            end
        end

        function y =brother(obj)
            if isempty(obj.father)
                y = {};
            else
                if obj.father.left == obj
                    y = obj.father.right;
                else
                    y = obj.father.left;
                end
            end
        end

        function [new_Left_idx,new_Right_idx]= predict(obj,x)
            switch obj.mark
                case -1
                    Left_idx = x(:,obj.bestFeature) < obj.bestPartValue;
                    Right_idx = x(:,obj.bestFeature) >= obj.bestPartValue;
                case 1
                    Left_idx = x(:,obj.bestFeature) > obj.bestPartValue;
                    Right_idx = x(:,obj.bestFeature) <= obj.bestPartValue;
            end
            new_Left_idx = Left_idx;
            new_Right_idx = Right_idx;
            if ~isempty(obj.left)
                [~,L_Right_idx]= obj.left.predict(x(Left_idx,:));
                tmp_left = new_Left_idx(Left_idx);
                tmp_left(L_Right_idx) = 0;
                new_Left_idx(Left_idx) = tmp_left;

                tmp_right = new_Right_idx(Left_idx);
                tmp_right(L_Right_idx) = 1;
                new_Right_idx(Left_idx) = tmp_right;
            end

            if ~isempty(obj.right)
                [R_Left_idx,~]= obj.right.predict(x(Right_idx,:));

                tmp_right = new_Right_idx(Right_idx);
                tmp_right(R_Left_idx) = 0;
                new_Right_idx(Right_idx) = tmp_right;

                tmp_left = new_Left_idx(Right_idx);
                tmp_left(R_Left_idx) = 1;
                new_Left_idx(Right_idx) = tmp_left;
            end

        end
    end
end

function [Ly,Ry ]= splitDataset(dataset,axis,value)
idx_L = dataset(:,axis)<value;
idx_R = dataset(:,axis)>=value;
Ly = dataset(idx_L,:);
Ry = dataset(idx_R,:);
end

function y = calcShannonEnt(dataset)
Y = dataset(:,end);
SET = unique(Y);
N = length(Y);
y = 0;
for i = 1:length(SET)
    number_i = sum(Y==SET(i));
    y = y + (number_i/N)*log2((number_i/N));
end
y = -y;
end


