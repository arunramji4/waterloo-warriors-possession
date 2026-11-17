def goalkeeper_ids(metadata):
    return {p.attrib['id'] for p in metadata.findall('.//Players/Player')
            if any(x.findtext('Name')=='position_type' and
                   (x.findtext('Value') or '').lower()=='goalkeeper'
                   for x in p.findall('.//ProviderParameter'))}
