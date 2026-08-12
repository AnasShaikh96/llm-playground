# Source - https://stackoverflow.com/a/67962816
# Posted by furas, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-12, License - CC BY-SA 4.0

import http.server
import socketserver
import json

PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_POST(self):
        # - request -
        
        content_length = int(self.headers['Content-Length'])
        #print('content_length:', content_length)
        
        if content_length:
            input_json = self.rfile.read(content_length)
            input_data = json.loads(input_json)
        else:
            input_data = None
            
        print(input_data)
        
        # - response -
        
        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.end_headers()
        
        output_data = {'status': 'OK', 'result': 'HELLO WORLD!'}
        output_json = json.dumps(output_data)
        
        self.wfile.write(output_json.encode('utf-8'))


Handler = MyHandler

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Starting http://0.0.0.0:{PORT}")
        httpd.serve_forever()
except KeyboardInterrupt:
    print("Stopping by Ctrl+C")
    httpd.server_close()  # to resolve problem `OSError: [Errno 98] Address already in use` 
