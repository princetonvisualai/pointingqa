// Misc utility functions that don't fit anywhere else.
var VG = (function(vg) {

  // For each property in default_options that is not in options, add that
  // property to options.
    //
  vg.addSuffix = function(names, suffix) {
      for (var i=0; i< names.length; i++){
          names[i] = String(names[i]) + suffix;
      }
  }

  vg.addUnderscore= function(names) {
      for (var i=0; i< names.length; i++){
          var name = String(names[i]);
          names[i] = name.substring(0,4) + '_' + name.substring(4,name.length);
      }
  }

  vg.drawCoordinates = function(x,y){
    var pointSize = 3; // Change according to the size of the point.
    var ctx = document.getElementById("canvas").getContext("2d");

    ctx.fillStyle = "#ff2626"; // Red color

    ctx.beginPath(); //Start path
    ctx.arc(x, y, pointSize, 0, Math.PI * 2, true); // Draw a point using the arc function of the canvas with a point structure.
    ctx.fill(); // Close the path and fill.

  }
  vg.stripExt = function(names) {
      var shortened = []
      var N = names.length;
      for(var i = 0; i < N; i++){
          shortened.push(names[i].slice(0,-4)); // Strip .jpg
      }
    return shortened;
  }

  vg.shuffle = function(arr, cb){
            for (var i = arr.length-1; i >= 0; i--){
                var ind = Math.floor(Math.random() * (i+1));
                var temp  = arr[ind];
                arr[ind] = arr[i];
                arr[i] = temp;
            }
           cb(arr); 
    }
  vg.reservoirSelect = function(img_data,batch,cb) {
            var images = [];
            for(var i = 0; i < img_data.length; i++){
                var filename = img_data[i]; //.Key
                if(i < batch) 
                    images.push(filename);
                else{
                    var r = Math.floor(Math.random()*i)
                    if(r < batch){
                        images[r] = filename;
                    }
                }
            }
            cb(images);
    }
  vg.merge_options = function(options, default_options) {
    if (typeof(options) === 'undefined') {
      options = {};
    }

    for (var opt in default_options) {
      if (default_options.hasOwnProperty(opt)
          && !options.hasOwnProperty(opt)) {
        options[opt] = default_options[opt];
      }
    }

    return options;
  }

  /**
   * Dynamically create a Bootstrap collapse widget from data.
   *
   * div - The div that will hold the collapse widget. It will be
   *       given a unique ID if it doesn't already have one.
   * data - List of the following form:
   *        [{'title': "A title", 'render': function(div) { }}, ... ]
   *        each element of the list corresponds to a panel in the div.
   *        The title gives the title of the panel, and the render
   *        callback will be passed the body of the panel.
   */
  vg.make_bootstrap_collapse = function(div, data) {
    div.addClass('panel-group');
    var div_id = div.attr('id');
    if (!div_id) {
      div_id = _.uniqueId();
      div.attr('id', div_id);
    }
    for (var i = 0; i < data.length; i++) {
      (function() {
        var panel = $('<div>').addClass('panel panel-default')
                              .appendTo(div);
        var heading = $('<div>').addClass('panel-heading')
                                .appendTo(panel);
        var title = $('<h4>').addClass('panel-title')
                             .appendTo(heading);
        var collapse_id = _.uniqueId();
        var collapse = $('<div>').addClass('panel-collapse collapse')
                                 .attr('id', collapse_id)
                                 .appendTo(panel);
        var body = $('<div>').addClass('panel-body')
                             .appendTo(collapse);
        var link = $('<a>').attr({'data-toggle': 'collapse',
                                  'data-parent': '#' + div_id,
                                  'href': '#' + collapse_id,})
                           .text(data[i].title)
                           .appendTo(title);
        var rendered = false;
        var render_fn = data[i].render;
        collapse.on('show.bs.collapse', function() {
          if (!rendered) {
            render_fn(body);
            rendered = true;
          }
        });
      })();
    }
  }

  /**
   * Dynamicalled create a list of images from data.
   *
   * div - The dic that will hold the image list.
   * data - a list of urls.
   *
   */
  vg.make_image_list = function(div, data) {
    for (var image_index in data) {
      var url = data[image_index];
      $('<img>').attr({'src': url, 'width': 100, 'height': 100, 'margin': '5px'})
                .appendTo(div);
    }
  }

  return vg;

}(VG || {}));
