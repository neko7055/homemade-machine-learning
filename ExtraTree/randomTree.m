classdef randomTree < handle
    %DECISION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        TF
        root
        max_depth

        
    end
    methods
        function obj = randomTree(max_depth)
            obj.root = ExtraTreeNode();
            if nargin==0
                if isempty(obj.max_depth)
                    obj.max_depth = 5;
                end
            else
                obj.max_depth = max_depth;
            end
        end
        
        function  fit(obj,x,y)
            if isempty(obj.TF)
                obj.TF = minimax_tf(x);
            end
            x = obj.TF.tf(x);
            nd = {{obj.root,x,y}};
            for i = 1:obj.max_depth
                new_nd = {};
                for j = 1:length(nd)
                nd{j}{1}.fit( nd{j}{2}, nd{j}{3});
                [Left_idx,Right_idx]= nd{j}{1}.predict(nd{j}{2});
                
                if (sum(nd{j}{3}(Left_idx)==-1)~=length(nd{j}{3}(Left_idx))) && (i~=obj.max_depth)
                    nd{j}{1}.left = ExtraTreeNode();
                    nd{j}{1}.left.father = nd{j}{1};
                    new_nd{end+1} = {nd{j}{1}.left, nd{j}{2}(Left_idx,:), nd{j}{3}(Left_idx,:)};
                    
                end
                if (sum(nd{j}{3}(Right_idx)==1)~=length(nd{j}{3}(Right_idx))) && (i~=obj.max_depth)
                    nd{j}{1}.right = ExtraTreeNode();
                    nd{j}{1}.right.father = nd{j}{1};
                    new_nd{end+1} = {nd{j}{1}.right, nd{j}{2}(Right_idx,:) ,nd{j}{3}(Right_idx,:)};
                end
                end
                nd = new_nd;
            end
        end
        function out = copy(obj)
            out = randomTree();
            if ~isempty(obj.TF)
                out.TF =obj.TF.copy();
            end
            out.root = obj.root.copy();
            out.max_depth = obj.max_depth;
        end
        function label = predict(obj,x)
            x = obj.TF.tf(x);
            [Left_idx,Right_idx] = obj.root.predict(x);
            label = Right_idx - Left_idx;
        end
    end
end


