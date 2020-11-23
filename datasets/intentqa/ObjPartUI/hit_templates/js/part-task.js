var VG = (function(vg) {
	//task_input must have:
	//  image, obj_name, obj_plural, gt.bbox, gt.min_num, gt.max_num
    vg.PartTask = function(container_div,task_input) {
        var that = (this === vg ? {} : this);
        var enabled = false;

        //var image_url = task_input.image_url;

        //var drawer_div = $('<div>').appendTo(container_div);

        VIEWPORT_HEIGHT = 500;
        that.canvas = $(task_input.canvas);

        var max_height = VIEWPORT_HEIGHT;
            max_height -= $('#task-instr-div').height();
            max_height -= $('#c-buttons-div').height();
        var images_div = $('#c-imgs-div');

        //var bbox_drawer = new VG.PointDrawer(drawer_div,image_url,null,bbox_drawer_options);
        var timer = new VG.Timer();
        that.GetAnswerIfValid = function () {
            var ans = task_input.getAnswer();
            var correct = task_input.getGT();
            if (ans != correct){            
                return -1; 
            }
            return { [ task_input.im_name+"_"+task_input.obj_id+"_"+task_input.inst_id+"_"+task_input.part_id+"_"+task_input.xCoord+"_"+task_input.yCoord ] : 
                    { 'answer': ans, 
                    //'im_name': task_input.im_name,
                    //'obj_id': task_input.obj_id,
                    //'inst_id': task_input.inst_id,
                    //'part_id': task_input.part_id,
                    //'xCoord': task_input.xCoord,
                    //'yCoord': task_input.yCoord,
                    //'total_parts': 'NA',
                    //'obj_area': 'NA',
                    'im_area': task_input.im_area, 
                    'part_area':task_input.part_area, 
                    'time': timer.total()
                 }
            }
        }


        that.enable = function() {
            enabled = true;
            that.canvas.show();
            //bbox_drawer.enable();
            timer.start();
        }

        that.disable = function() {
            enabled = false;
            //bbox_drawer.disable();
            that.canvas.hide();
            timer.stop();
        }
        
        that.disable();
        
        return that;
    }
    
  return vg;

}(VG || {}));
