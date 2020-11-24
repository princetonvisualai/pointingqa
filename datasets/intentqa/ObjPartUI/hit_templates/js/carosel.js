var VG = (function(vg) {

  /*
   * A Carosel allows different panes of content to be shown. Buttons and
   * keyboard shortcuts are used to navigate between panes.
   *
   * container_div: Parent div; for each pane a new div will be created
   *                inside this parent div.
   * button_div: A pair of buttons and a span will be added to this div for
   *             navigating between panes.
   * options_div: Stores 4 buttons to respond to the image,point prompt
   * num_panes: The number of panes to be shown in this carosel
   * show_callback(idx, div): A callback function called whenever a pane
   *                          becomes active. idx is the numeric index of the
   *                          pane being activated, and div is the div for the
   *                          newly activated pane.
   * should_scroll: Boolean, optional.
   *                If true, when the buttons are clicked an additional
   *                callback function is passed to show_callback. The show_callback
   *                can call this callback when it is done rendering to scroll the
   *                page back to the position that it was in prior to clicking any
   *                buttons. This is useful if rendering modifies the page length
   *                and is not instanteanous (such as if images are loaded).
   *
   * A Carosel has the following methods:
   * select_pane(new_idx): Set idx to be the current pane.
   * enable() / disable(): Enable or disable the carosel. It starts disabled.
   * enableKeyboardShortcuts() / disableKeyboardShortcuts(): Enable or disable
   *   the keyboard shortcuts. They start enabled, but only work if the carosel
   *   as a whole is also enabled.
   */

  vg.Carosel = function(canvases,point_canvases, id2text, container_div, button_div, options_div, num_panes,
                        show_callback, should_scroll) {
    var that = (this === vg ? {} : this);

      // Return all canvases
      //var canvases = container_div[0].getElementsByTagName('canvas');

    var enabled = false;
    if (typeof(should_scroll) === 'undefined') should_scroll = false;
    
    //Create onClick toggles
    var part = function(part_button)            {if (enabled) toggle_button(part_button);} 
    var entObj = function(obj_button)           {if (enabled) toggle_button(obj_button);};
    var imp2Tell = function(impossible_button)  {if (enabled) toggle_button(impossible_button);};
    var notPresent = function(na_button)        {if (enabled) toggle_button(na_button);};

    // First declare yucky global variables 
    var ans_buttons, part_button, obj_button, impossible_button, na_button, activeCanvas, activePointCanvas, bNextEnabled;

    // Create buttons
    var prev_button = $('<button>').prop('disabled', true)
                                   .text('Previous (left arrow)')
                                   .addClass('btn btn-default btn-lg padded')
                                   .appendTo(button_div);
    var counter_span = $('<span>').addClass('h3 padded vcenter init')
                                  .appendTo(button_div);
    var next_button = $('<button>').prop('disabled', true)
                                   .text('Next (right arrow)')
                                   .addClass('btn btn-default btn-lg padded')
                                   .appendTo(button_div);

    //Return the response for the image in activeCanvas
    function getResponse() {
      for(var i=0; i < ans_buttons.length; i++) {
            var b = ans_buttons[i]; 
            if(b.getAttribute('data-active') == 'true') { return b.id;}
      }
        return 'NA';
    }

    //Enable or disable the next button
    function enableNext() {
        next_button.prop('disabled', false);
        bNextEnabled = true;
    }
    function disableNext() {
        next_button.prop('disabled', true);
        bNextEnabled = false;
    }

    //Disable next, prev, and options
    function disable_buttons() {
      prev_button.prop('disabled', true);
      next_button.prop('disabled', true);
      for(var i=0; i < ans_buttons.length; i++) {
          var b = ans_buttons[i];
          $(b).prop('disabled', true);
      }
    }
    // Disable all but the previous button
    function deactivate_buttons() {
        disableNext();
        response = "";
        for(var i=0; i< ans_buttons.length; i++){
            var b = ans_buttons[i];
            turnOff(b);
        }
    }
      function toTitleCase(str)
        {
            return str.replace(/\w\S*/g, function(txt){return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();});
        }

    //Create a set of divs for the option buttons
    var optDivs = [];
    for (var i = 0; i < num_panes; i++) {
        activeCanvas = canvases[i]; 
        activePointCanvas = point_canvases[i];

        var objId = $(activeCanvas).attr('data-obj'); 
        var partId = $(activeCanvas).attr('data-part');
        var objText = id2text[objId]['name']; 
        //var partText = id2text[objId]['parts'][partId];
        var partText = toTitleCase(objText + " part [a]");
        //objText = toTitleCase('Whole ' + objText + ' [s]');
        if(typeof(partText)=== "undefined"){
            console.log('Part not found. Id number: ' + partId);
            console.log('Object id and number: ' + objId + ' - ' + objText);
            console.log(id2text[objId]);
        }

        var oDiv = $('<div class=buttons-container>').appendTo(options_div).hide();
        optDivs.push(oDiv);
        part_button = $('<button>').prop('disabled', true)
                                       .text(partText)
                                       .attr('id', String(partId))
                                       .addClass('btn btn-default btn-lg padded')
                                       .appendTo(oDiv);
        obj_button = $('<button>').prop('disabled', true)
                                       .text(toTitleCase('Whole ' + objText + ' [s]'))
                                       .attr('id', String(objId))
                                       .addClass('btn btn-default btn-lg padded')
                                       .appendTo(oDiv);
        impossible_button = $('<button>').prop('disabled', true)
                                       .text('Impossible to Tell [d]')
                                       .attr('id', '-1')
                                       .addClass('btn btn-default btn-lg padded')
                                       .appendTo(oDiv);
        na_button = $('<button>').prop('disabled', true)
                                       .text(toTitleCase('Not ' + objText + ' [f]'))
                                       .attr('id', '-2')
                                       .addClass('btn btn-default btn-lg padded')
                                       .appendTo(oDiv);
        part_button[0].onclick = function() { if (enabled) toggle_button(part_button);};
        obj_button[0].onclick = function() { if (enabled)  toggle_button(obj_button);};
        impossible_button[0].onclick = function() {  if (enabled)  toggle_button(impossible_button);};
        na_button[0].onclick = function() {  if (enabled)  toggle_button(na_button);};
        ans_buttons = [part_button[0], obj_button[0], impossible_button[0], na_button[0]];

        deactivate_buttons();
    }

    // Response choice
    var response = "";

    var current_idx = 0;

    // Handlers for selecting next or previous pane
    var next = function() { if (enabled && bNextEnabled) select_pane(current_idx + 1, should_scroll); };
    var prev = function() { if (enabled) select_pane(current_idx - 1, should_scroll); };
    var toggleColor = function(canvas) { 
        if (canvas.classList.contains("color")) {
            canvas.classList.remove("color");
            canvas.classList.add("grayscale");
        } else{
            canvas.classList.remove("grayscale");
            canvas.classList.add("color");
        }
    }

	     
    next_button.click(next);
    prev_button.click(prev);

    var keyboard_enabled = true;

    // Set up keypress handlers for keyboard shortcuts
    //prevKeys = [37, 65]; // a and left
    //nextKeys = [39, 68]; // d and right
    prevKeys = [37, 72]; // left and h
    nextKeys = [39, 76]; // right and l
    partKeys = [65];
    objKeys = [83]; // s is whole object
    impKeys = [68];// d is impossible to tell
    naKeys = [70]; // f is none of the above
    colorKeys = [84]; // f is none of the above
    $(document.documentElement).keyup(function(e) {
	    if (!keyboard_enabled || !enabled) return;
	    if ($.inArray(e.keyCode, prevKeys) !== -1) prev();
	    if ($.inArray(e.keyCode, nextKeys) !== -1) next();
	    if ($.inArray(e.keyCode, partKeys) !== -1) part(part_button); 
	    if ($.inArray(e.keyCode, objKeys) !== -1) entObj(obj_button); 
	    if ($.inArray(e.keyCode, impKeys) !== -1) imp2Tell(impossible_button); 
	    if ($.inArray(e.keyCode, naKeys) !== -1) notPresent(na_button); 
	    if ($.inArray(e.keyCode, colorKeys) !== -1) toggleColor(activeCanvas); 

	});

    that.enable = function() {
      enable_buttons();
      enabled = true;
    }

    that.disable = function() {
      disable_buttons();
      enabled = false;
    }

    that.enableKeyboardShortcuts = function() { keyboard_enabled = true; };
    that.disableKeyboardShortcuts = function() { keyboard_enabled = false; };

    function select_pane(new_idx, scroll) {
      // Ignore out-of-bounds calls
      if (new_idx < 0 || new_idx >= num_panes) return;

      if (typeof(scroll) === 'undefined') {
        scroll = false;
      }

      var scroll_pos = $('body').scrollTop();
      function cb() {
        if (scroll) $('body').scrollTop(scroll_pos);
      }

      // Hide all canvases 
      for (var i = 0; i < canvases.length; i++) {
        $(canvases[i]).hide();
        $(canvases[i]).parent().hide();
        $(point_canvases[i]).hide();
        optDivs[i].hide();
      }

      // Show the proper div and call the callback
      $(canvases[new_idx]).show();
      $(canvases[new_idx]).parent().show();
      $(point_canvases[new_idx]).show();
      //$(canvases[new_idx]).parentElement.show();
    
      optDivs[new_idx].show();
      activeCanvas = canvases[new_idx];
      activePointCanvas = point_canvases[new_idx];

      //Expose buttons
      var optDiv = optDivs[new_idx][0];
      ans_buttons = optDiv.getElementsByTagName('button');

      part_button = ans_buttons[0];
      obj_button = ans_buttons[1];
      impossible_button = ans_buttons[2];
      na_button = ans_buttons[3];
      if (scroll) {
        show_callback(new_idx, canvases[new_idx], cb);
      } else {
        show_callback(new_idx, canvases[new_idx]);
      }

      // Update the text of the counter
      counter_span.text((new_idx + 1) + ' / ' + num_panes);

      current_idx = new_idx;

      // Only allow forward progress once an answer is selected
      if (enabled) enable_buttons();
    }
    that.select_pane = select_pane;

    function enable_buttons() {
      disable_buttons();
      // Enable / disable buttons depending on index
      if (current_idx > 0) {
        prev_button.prop('disabled', false);
      }
      var response = getResponse();
      if(response == 'NA'){ disableNext();}
      else {enableNext();}
      //if (current_idx !== num_panes - 1) {
      //  next_button.prop('disabled', false);
      //}
      for(var i=0; i < ans_buttons.length; i++) {
          var b = ans_buttons[i];
          $(b).prop('disabled', false);
      }
      
    }

    //highlight button
    function turnOn(b) {
        b.setAttribute('data-active', 'true');
        response = b.id;
        b.style.background='#34A7C1';
        b.style.text='#fff';
        activeCanvas.setAttribute('data-response', response);
    }
    //unhighlight button
    function turnOff(b) {
        b.setAttribute('data-active', 'false');
        activeCanvas.setAttribute('data-response', 'NA');
        b.style.background='#fff';
        b.style.text='#333';
    }

    function select_button(b) {
        deactivate_buttons();
        turnOn(b);
        enableNext();
    }
    function toggle_button(b) {
        if (b.getAttribute('data-active') == 'false'){
            select_button(b);
        } else {deactivate_buttons();}
    }

    select_pane(0, false);
    deactivate_buttons();
    return that;
  }

  return vg;

}(VG || {}));
