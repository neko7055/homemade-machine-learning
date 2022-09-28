classdef decision < handle
    %DECISION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        TF
        x
        i
        threshold
        mark
        y
        W
        minE
        
    end
    methods
        function obj = decision()
        end
        
        function  fit(obj,x,y,W,~)
            switch nargin
                case 3
                    if isempty(obj.W)
                        obj.W = ones(size(x,1),1)/size(x,1);
                    end
                otherwise
                    obj.W = W;
            end
            if isempty(obj.TF)
                obj.TF = minimax_tf(x);
            end
            obj.x = obj.TF.tf(x);
            obj.y = y;
            [~,M] = size(obj.x);
            obj.minE=-inf;
            for i = 1:M
                    data = obj.x(:,i);
                    uniqueVals = unique(data);
                    sortedUniqueVals = sort(uniqueVals);
                    for j = 1:length(sortedUniqueVals)-1
                        threshold = (sortedUniqueVals(j) + sortedUniqueVals(j+1))/2;
                        for mark = [-1,1]
                            sum_error = (obj.W' *(classify(obj.x,i,threshold,mark).*obj.y));
                            if (sum_error) > obj.minE
                                obj.minE = sum_error;
                                obj.i=i;
                                obj.threshold=threshold;
                                obj.mark=mark;
                            end
                        end
                    end
            end
        end
        function out = copy(obj)
                out = decision();
                out.x =obj.x;
                out.i =obj.i;
                out.threshold =obj.threshold; 
                out.mark =obj.mark;
                out.y =obj.y;
                out.W = obj.W;
                out.minE =obj.minE;
                if ~isempty(obj.TF)
                    out.TF =obj.TF.copy();
                end
        end
        function label = predict(obj,x)
            x = obj.TF.tf(x);
            label = classify(x,obj.i,obj.threshold,obj.mark);
        end
    end
end
function label = classify(x,i,threshold,mark)
    label = mark*((x(:,i) <= threshold) - (x(:,i) > threshold));
end
