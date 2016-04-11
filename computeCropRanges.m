function [left_range,top_range] = computeCropRanges(result_bbox,w,h,crop_w,crop_h)

%compute margins, which tell us how much wiggle room there is to move the bounding box of the moth
min_r = result_bbox(1); max_r = result_bbox(2);
min_c = result_bbox(3); max_c = result_bbox(4);

bbox_height = max_r - min_r + 1;
bbox_width = max_c - min_c + 1;

left_margin = crop_w - bbox_width;
top_margin = crop_h - bbox_height;

left_range = [max(1,min_c - left_margin), min_c];
top_range = [max(1,min_r - top_margin), min_r];

right_limit = w-crop_w;
bottom_limit = h-crop_h;

if(left_range(1) > right_limit)
    display('Error');
    return;
end

if(top_range(1) > bottom_limit)
    display('Error');
    return;
end

left_range(2) = min(left_range(2),right_limit); %can't have crop go off image
top_range(2) = min(top_range(2),bottom_limit);

end