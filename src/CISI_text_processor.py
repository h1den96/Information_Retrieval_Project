import json

def parse_cisi_file(file_path):
    documents = []
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into individual documents by .I delimiter
    raw_documents = content.split('.I ')[1:]

    for raw_doc in raw_documents:
        # Extract the ID (appears at the beginning)
        id_split = raw_doc.split('\n', 1)
        doc_id = int(id_split[0].strip())
        remaining_content = id_split[1] if len(id_split) > 1 else ""

        # Initialize title and content
        title, content = "", ""

        # Extract .T section (Title)
        if '.T\n' in remaining_content:
            title_split = remaining_content.split('.T\n', 1)
            title_part = title_split[1] if len(title_split) > 1 else ""
            title = title_part.split('\n.', 1)[0].strip() if '\n.' in title_part else title_part.strip()
            remaining_content = title_split[1].split('\n.', 1)[1] if '\n.' in title_split[1] else ""

        # Extract .W section (Content)
        if '.W\n' in remaining_content:
            content_split = remaining_content.split('.W\n', 1)
            content_part = content_split[1] if len(content_split) > 1 else ""
            content = content_part.split('\n.', 1)[0].strip() if '\n.' in content_part else content_part.strip()

        # Add parsed document to the list
        documents.append({
            "id": doc_id,
            "title": title,
            "content": content
        })

    return documents

# Input and Output Paths
input_file = './CISI.ALL'
output_file = './CISI_articles.json'

# Parse the file and convert to JSON
parsed_documents = parse_cisi_file(input_file)

# Write to JSON file
with open(output_file, 'w') as json_file:
    json.dump(parsed_documents, json_file, indent=2)

print(f"Converted CISI.ALL to JSON and saved to {output_file}")
