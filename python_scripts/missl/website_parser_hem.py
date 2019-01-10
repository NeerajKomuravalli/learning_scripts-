import os


if __name__ == '__main__':
    folder_path = "webistes/"
    list_of_files = os.listdir(folder_path)
    add_if_host_line = ['server {\n', '    if ($host = %s) {\n', '                return 301 http://www.$host$request_uri;\n', '        }\n']
    server_two_code = ['server {\n', '    if ($host = www.%s) {\n', '        return 301 https://$host$request_uri;\n', '    } # managed by Certbot\n', '    if ($host = %s) {\n', '        return 301 https://$host$request_uri;\n', '    } # managed by Certbot\n', '    server_name %s  www.%s;\n', '    listen 80;\n', '    return 404; # managed by Certbot\n', '}\n']

    for file in list_of_files:
        if file == "venuenetwork.info":
            continue
        with open(os.path.join(folder_path, file)) as f:
            lines = f.readlines()
            lines = list(filter(lambda x: x != "\n", lines))
            if_host_present = "if ($host" in lines[1]
            ssl_present = list(filter(lambda x: "listen 443 ssl" in x, lines)) != []

            server_code_no = len(list(filter(lambda x: x == 'server {\n', lines)))
            if server_code_no == 1 and ssl_present:
                lines.append("".join(server_two_code)%(file, file, file, file))

            if not if_host_present:
                lines[0] = "".join(add_if_host_line)%(file)

            with open(os.path.join(folder_path, file), 'w') as new_f:
                new_f.write("".join(lines))
                new_f.close()