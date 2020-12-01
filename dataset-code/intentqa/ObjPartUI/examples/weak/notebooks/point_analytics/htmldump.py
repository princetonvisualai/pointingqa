'''Used for debugging and displaying images from results.'''


def create_html_file(image_data, title="HTML Dump"):
    import base64
    from io import BytesIO

    def create_gallery_object(entry):
        html = '''<div class="gallery">
          <a target="_blank" >'''
        buffered = BytesIO()
        entry[2].save(buffered, format="JPEG")
        data_uri = base64.b64encode(buffered.getvalue()).replace('\n', '')
        # data_uri = ambiguous_imgs[0][1].encode('base64').replace('\n', '')
        img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
        html += img_tag
        html += "</a>"
        html += "<div class=\"desc\">{}\n{}</div>".format(entry[0], entry[1])
        html += "</div>"
        return html

    html = ''' <html>
        <head><title>'''
    html += title
    html += '''
        </title>
        <style>
        div.gallery {
            margin: 5px;
            border: 1px solid #ccc;
            float: left;
            width: 300px;
        }

        div.gallery:hover {
            border: 1px solid #777;
        }

        div.gallery img {
            width: 100%;
            height: auto;
        }

        div.desc {
            padding: 15px;
            text-align: center;
        }
        </style>
        </head>
        <body>
        '''
    for entry in image_data:
        html += create_gallery_object(entry)

    html += '''
            </body>
            </html>'''

    return html
