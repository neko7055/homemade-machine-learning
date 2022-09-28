classdef minimax_tf < handle
    %MINIMAX_TF Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        min_v
        range
        enable
    end
    
    methods
        function obj = minimax_tf(data)
            if nargin == 0
                return
            end
            obj.min_v = min(data,[],1);
            obj.range = max(data,[],1) - obj.min_v;
            if isempty(obj.enable)
                obj.enable = true;
            end
        end
        function out = copy(obj)
            out = minimax_tf();
            out.min_v = obj.min_v;
            out.range = obj.range;
            out.enable = obj.enable;
        end
        
        function y =tf(obj,x)
            if obj.enable
                y = (x - obj.min_v)./obj.range;
            else
                y = x;
            end
        end
    end
end

