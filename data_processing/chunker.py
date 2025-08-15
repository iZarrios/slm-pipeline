from typing import List


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Chunks text into smaller pieces of a specified size. Respect code blocks and paragraphs.
    (this is mainly targeting markdown files)

    Args:
        text: The text to chunk
        chunk_size: The size of each chunk (default is 1000 characters)

    Returns:
        List[str]: List of text chunks, with code blocks reserved as single chunks
    """

    chunks = []
    chunk_start = 0
    text_length = len(text)
    in_code_block = False

    while chunk_start < text_length:
        # calc position
        chunk_end = chunk_start + chunk_size

        if chunk_end >= text_length:
            chunks.append(text[chunk_start:].strip())
            break

        chunk_block = text[chunk_start:chunk_end]

        # handling code blocks

        code_block_start = chunk_block.find("```")
        if code_block_start != -1:
            if not in_code_block:
                code_block_end = text.find("```", chunk_start + code_block_start + 3)

                if code_block_end != -1:
                    # include the entire code block as one chunk
                    chunk = text[chunk_start : code_block_end + 3].strip()

                    if chunk:
                        chunks.append(chunk)
                    chunk_start = code_block_end + 3
                    continue
                else:
                    in_code_block = True
                    chunk_end = chunk_start + code_block_start
            else:
                in_code_block = False
                chunk_end = chunk_start + code_block_start + 3

        # if we are in a code block, continue the next iteration to find its end
        if in_code_block:
            chunk_end = min(chunk_end, text_length)
            chunks.append(text[chunk_start:chunk_end].strip())
            chunk_start = chunk_end
            continue
        # if no code blocks we try to find a paragraph
        if "\n\n" in chunk_block:
            # find the end of the paragraph
            last_break = chunk_block.find("\n\n")
            if last_break > chunk_size * 0.3:  # break past 30% of the chunk
                chunk_end = chunk_start + last_break

        # break at sentence if no code block or paragraph
        elif ". " in chunk_block:
            last_period = chunk_block.rfind(". ")
            if last_period > chunk_size * 0.3:  # break past 30% of the chunk
                chunk_end = chunk_start + last_period + 1

        # clean up the chunk
        chunk = text[chunk_start:chunk_end].strip()
        if chunk:
            chunks.append(chunk)

        # update the chunk start
        chunk_start = max(chunk_start + 1, chunk_end)

    return chunks
